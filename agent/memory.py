# agent/memory.py — Memoria de conversaciones con SQLite
# Generado por AgentKit para Sucol Soluciones Urbanísticas

"""
Sistema de memoria de Sofía. Guarda el historial de conversaciones
por número de teléfono usando SQLite (local) o PostgreSQL (producción).
"""

import asyncio
import logging
import os
import ssl
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, select, Integer
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("agentkit")

# Credenciales Supabase CRM (leads-sucol) para sincronizar conversaciones
_SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
_SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# Configuración de base de datos
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./agentkit.db")

# Remover ?sslmode=require si viene en la URL (asyncpg no lo soporta como parámetro)
if "?sslmode=" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split("?sslmode=")[0]

# Si es PostgreSQL en producción, ajustar el esquema de URL
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

# SSL para Supabase: cifrado sin verificar certificado (pooler usa cert autofirmado)
is_postgres = DATABASE_URL.startswith("postgresql+asyncpg://")
if is_postgres:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    connect_args = {"ssl": ssl_ctx}
else:
    connect_args = {}

engine = create_async_engine(DATABASE_URL, echo=False, connect_args=connect_args)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Mensaje(Base):
    """Modelo de mensaje en la base de datos."""
    __tablename__ = "mensajes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telefono: Mapped[str] = mapped_column(String(50), index=True)
    role: Mapped[str] = mapped_column(String(20))  # "user" o "assistant"
    content: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


async def inicializar_db():
    """Crea las tablas si no existen."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _sync_a_supabase(telefono: str, role: str, content: str, timestamp: datetime) -> None:
    """Sincroniza un mensaje a conversaciones_sofia en Supabase. Fire-and-forget."""
    if not _SUPABASE_URL or not _SUPABASE_SERVICE_KEY:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{_SUPABASE_URL}/rest/v1/conversaciones_sofia",
                headers={
                    "apikey": _SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {_SUPABASE_SERVICE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json={
                    "telefono": telefono,
                    "role": role,
                    "content": content,
                    "timestamp": timestamp.isoformat(),
                },
            )
            if resp.status_code not in (200, 201):
                logger.warning(f"[Supabase sync] HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as exc:
        logger.warning(f"[Supabase sync] Error al sincronizar mensaje de {telefono}: {exc}")


async def guardar_mensaje(telefono: str, role: str, content: str):
    """Guarda un mensaje en el historial de conversación y lo sincroniza a Supabase."""
    ts = datetime.utcnow()
    async with async_session() as session:
        mensaje = Mensaje(
            telefono=telefono,
            role=role,
            content=content,
            timestamp=ts,
        )
        session.add(mensaje)
        await session.commit()
    # Sincronización async a Supabase — no bloquea, fallo silencioso
    asyncio.create_task(_sync_a_supabase(telefono, role, content, ts))


def _variantes_telefono_memoria(telefono: str) -> list[str]:
    """Genera variantes para encontrar historial aunque cambie el formato."""
    t = telefono.strip().replace(" ", "").replace("-", "")
    variantes = {t}
    if not t.startswith("+"):
        variantes.add("+" + t)
    if t.startswith("57") and len(t) == 12:
        variantes.add(t[2:])
    if t.startswith("+57") and len(t) == 13:
        variantes.add(t[3:])
    return list(variantes)


async def _obtener_historial_supabase(telefono: str, limite: int) -> list[dict]:
    """Lee historial desde conversaciones_sofia en Supabase."""
    if not _SUPABASE_URL or not _SUPABASE_SERVICE_KEY:
        return []

    variantes = _variantes_telefono_memoria(telefono)
    filtro_or = ",".join(f"telefono.eq.{v}" for v in variantes)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{_SUPABASE_URL}/rest/v1/conversaciones_sofia",
                headers={
                    "apikey": _SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {_SUPABASE_SERVICE_KEY}",
                },
                params={
                    "select": "role,content,timestamp,created_at",
                    "or": f"({filtro_or})",
                    "order": "timestamp.desc.nullslast,created_at.desc",
                    "limit": str(limite),
                },
            )
            if resp.status_code != 200:
                logger.warning(f"[Supabase historial] HTTP {resp.status_code}: {resp.text[:120]}")
                return []
            rows = resp.json()
    except Exception as exc:
        logger.warning(f"[Supabase historial] Error al leer historial de {telefono}: {exc}")
        return []

    rows.reverse()
    historial = []
    for row in rows:
        role = row.get("role")
        content = row.get("content")
        if role in ("user", "assistant") and content:
            historial.append({"role": role, "content": content})

    if historial:
        logger.info(f"[Supabase historial] {telefono}: {len(historial)} mensajes recuperados")
    return historial


async def obtener_historial(telefono: str, limite: int | None = None) -> list[dict]:
    """
    Recupera los últimos N mensajes de una conversación.
    """
    limite = limite or int(os.getenv("HISTORIAL_LIMITE", "100"))

    historial_supabase = await _obtener_historial_supabase(telefono, limite)
    if historial_supabase:
        return historial_supabase

    async with async_session() as session:
        query = (
            select(Mensaje)
            .where(Mensaje.telefono == telefono)
            .order_by(Mensaje.timestamp.desc())
            .limit(limite)
        )
        result = await session.execute(query)
        mensajes = result.scalars().all()
        mensajes.reverse()
        return [
            {"role": msg.role, "content": msg.content}
            for msg in mensajes
        ]


async def limpiar_historial(telefono: str):
    """Borra todo el historial de una conversación."""
    async with async_session() as session:
        query = select(Mensaje).where(Mensaje.telefono == telefono)
        result = await session.execute(query)
        mensajes = result.scalars().all()
        for msg in mensajes:
            await session.delete(msg)
        await session.commit()
