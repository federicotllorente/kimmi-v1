"""Crea estructuras mínimas de proyecto.

Plantillas: nextjs, micropython, nodejs, python, fastapi, esp32.
Solo escribe dentro de allowed_root. Sin git. open_in: cursor|vscode.
"""
from __future__ import annotations

import os
import re
from typing import List, Tuple

from tools.sandbox import resolve_safe_path

TEMPLATES = ("nextjs", "micropython", "nodejs", "python", "fastapi", "esp32")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def scaffold_project(
    template: str,
    name: str,
    allowed_root: str,
    parent_path: str = ".",
    open_in: str = "",
) -> str:
    """Crea un proyecto desde plantilla embebida bajo el sandbox."""
    template = (template or "").strip().lower()
    name = (name or "").strip()
    open_in = (open_in or "").strip().lower()

    if template not in TEMPLATES:
        return f"Plantilla no válida. Usa: {', '.join(TEMPLATES)}."

    if open_in and open_in not in ("cursor", "vscode"):
        return "open_in debe ser '' , 'cursor' o 'vscode'."

    if not name or not _NAME_RE.match(name):
        return (
            "Nombre de proyecto inválido. Usa letras, números, puntos, "
            "guiones o guiones bajos (máx. 64)."
        )

    parent_safe, error = resolve_safe_path(parent_path or ".", allowed_root)
    if error:
        return error
    if not os.path.isdir(parent_safe):
        return f"El directorio padre '{parent_path}' no existe."

    dest = os.path.join(parent_safe, name)
    dest_safe, dest_error = resolve_safe_path(dest, allowed_root)
    if dest_error:
        return dest_error

    if os.path.exists(dest_safe):
        if os.path.isdir(dest_safe) and os.listdir(dest_safe):
            return f"El directorio '{name}' ya existe y no está vacío."
        if os.path.isfile(dest_safe):
            return f"Ya existe un archivo llamado '{name}'."

    files = _template_files(template, name)
    created: List[str] = []

    try:
        for rel, content in files:
            full = os.path.join(dest_safe, rel.replace("/", os.sep))
            full_safe, full_err = resolve_safe_path(full, allowed_root)
            if full_err:
                return full_err
            parent = os.path.dirname(full_safe)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(full_safe, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            created.append(rel)
    except Exception as e:
        return f"No se pudo crear el proyecto: {e}"

    root = os.path.realpath(allowed_root)
    rel_root = os.path.relpath(dest_safe, root)
    tree = "\n".join(f"  - {p}" for p in created)
    msg = f"Proyecto '{template}' creado en {rel_root}:\n{tree}"

    if open_in:
        from tools.launch_app import launch_app

        open_msg = launch_app(open_in, path=dest_safe, focus_after=True)
        msg = f"{msg}\n{open_msg}"

    return msg


def _template_files(template: str, name: str) -> List[Tuple[str, str]]:
    if template == "python":
        return [
            ("README.md", f"# {name}\n\nProyecto Python generado por Jarvis.\n"),
            (
                "main.py",
                (
                    f'def main() -> None:\n    print("Hello from {name}")\n\n\n'
                    'if __name__ == "__main__":\n    main()\n'
                ),
            ),
            ("requirements.txt", ""),
            (
                ".gitignore",
                "__pycache__/\n*.py[cod]\n.venv/\nvenv/\n.env\n*.egg-info/\ndist/\nbuild/\n",
            ),
        ]

    if template == "fastapi":
        return [
            ("README.md", f"# {name}\n\nAPI FastAPI generada por Jarvis.\n\n```bash\npip install -r requirements.txt\nuvicorn main:app --reload\n```\n"),
            (
                "main.py",
                (
                    "from fastapi import FastAPI\n\n"
                    "app = FastAPI(title=" + repr(name) + ")\n\n"
                    '@app.get("/")\ndef root():\n'
                    '    return {"message": "Hello from ' + name + '"}\n\n'
                    '@app.get("/health")\ndef health():\n'
                    '    return {"status": "ok"}\n'
                ),
            ),
            ("requirements.txt", "fastapi>=0.115.0\nuvicorn[standard]>=0.30.0\n"),
            (
                ".gitignore",
                "__pycache__/\n*.py[cod]\n.venv/\nvenv/\n.env\n",
            ),
        ]

    if template == "nodejs":
        return [
            (
                "package.json",
                (
                    "{\n"
                    f'  "name": "{name}",\n'
                    '  "version": "1.0.0",\n'
                    '  "private": true,\n'
                    '  "main": "index.js",\n'
                    '  "scripts": { "start": "node index.js" }\n'
                    "}\n"
                ),
            ),
            ("index.js", f'console.log("Hello from {name}");\n'),
            ("README.md", f"# {name}\n\nProyecto Node.js generado por Jarvis.\n"),
            (".gitignore", "node_modules/\n.env\nnpm-debug.log*\n"),
        ]

    if template == "nextjs":
        return [
            (
                "package.json",
                (
                    "{\n"
                    f'  "name": "{name}",\n'
                    '  "version": "0.1.0",\n'
                    '  "private": true,\n'
                    '  "scripts": {\n'
                    '    "dev": "next dev",\n'
                    '    "build": "next build",\n'
                    '    "start": "next start"\n'
                    "  },\n"
                    '  "dependencies": {\n'
                    '    "next": "^15.0.0",\n'
                    '    "react": "^19.0.0",\n'
                    '    "react-dom": "^19.0.0"\n'
                    "  },\n"
                    '  "devDependencies": {\n'
                    '    "@types/node": "^22.0.0",\n'
                    '    "@types/react": "^19.0.0",\n'
                    '    "@types/react-dom": "^19.0.0",\n'
                    '    "typescript": "^5.0.0"\n'
                    "  }\n"
                    "}\n"
                ),
            ),
            (
                "tsconfig.json",
                (
                    "{\n"
                    '  "compilerOptions": {\n'
                    '    "target": "ES2017",\n'
                    '    "lib": ["dom", "dom.iterable", "esnext"],\n'
                    '    "allowJs": true,\n'
                    '    "skipLibCheck": true,\n'
                    '    "strict": true,\n'
                    '    "noEmit": true,\n'
                    '    "esModuleInterop": true,\n'
                    '    "module": "esnext",\n'
                    '    "moduleResolution": "bundler",\n'
                    '    "resolveJsonModule": true,\n'
                    '    "isolatedModules": true,\n'
                    '    "jsx": "preserve",\n'
                    '    "incremental": true,\n'
                    '    "plugins": [{ "name": "next" }],\n'
                    '    "paths": { "@/*": ["./*"] }\n'
                    "  },\n"
                    '  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],\n'
                    '  "exclude": ["node_modules"]\n'
                    "}\n"
                ),
            ),
            (
                "next.config.js",
                "/** @type {import('next').NextConfig} */\nconst nextConfig = {};\nmodule.exports = nextConfig;\n",
            ),
            (
                "app/layout.tsx",
                (
                    "export const metadata = {\n"
                    f'  title: "{name}",\n'
                    '  description: "Generated by Jarvis",\n'
                    "};\n\n"
                    "export default function RootLayout({\n"
                    "  children,\n"
                    "}: {\n"
                    "  children: React.ReactNode;\n"
                    "}) {\n"
                    "  return (\n"
                    '    <html lang="en">\n'
                    "      <body>{children}</body>\n"
                    "    </html>\n"
                    "  );\n"
                    "}\n"
                ),
            ),
            (
                "app/page.tsx",
                (
                    "export default function Home() {\n"
                    f"  return <main><h1>{name}</h1></main>;\n"
                    "}\n"
                ),
            ),
            (
                "README.md",
                f"# {name}\n\nProyecto Next.js (App Router) generado por Jarvis.\n\n```bash\nnpm install\nnpm run dev\n```\n",
            ),
            (".gitignore", "node_modules/\n.next/\nout/\n.env*\n"),
        ]

    if template == "esp32":
        return [
            (
                "boot.py",
                (
                    "# ESP32 MicroPython boot\n"
                    "import gc\n"
                    "gc.collect()\n"
                ),
            ),
            (
                "main.py",
                (
                    "from machine import Pin\n"
                    "import time\n\n"
                    "led = Pin(2, Pin.OUT)\n\n"
                    "def main():\n"
                    f'    print("Hello from {name} on ESP32")\n'
                    "    while True:\n"
                    "        led.value(1)\n"
                    "        time.sleep(0.5)\n"
                    "        led.value(0)\n"
                    "        time.sleep(0.5)\n\n"
                    'if __name__ == "__main__":\n'
                    "    main()\n"
                ),
            ),
            ("lib/.gitkeep", ""),
            (
                "README.md",
                f"# {name}\n\nProyecto MicroPython para ESP32 generado por Jarvis.\n",
            ),
        ]

    # micropython
    return [
        (
            "main.py",
            (
                "def main():\n"
                f'    print("Hello from {name} (MicroPython)")\n\n'
                'if __name__ == "__main__":\n'
                "    main()\n"
            ),
        ),
        ("boot.py", "# Runs on every boot before main.py\n"),
        ("README.md", f"# {name}\n\nProyecto MicroPython generado por Jarvis.\n"),
        ("lib/.gitkeep", ""),
    ]
