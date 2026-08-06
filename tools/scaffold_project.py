"""Crea estructuras mínimas de proyecto (Next.js, MicroPython, Node.js, Python).

Solo escribe dentro de allowed_root (sandbox). Sin npx/npm create.
Requiere: stdlib + tools.sandbox.
"""
from __future__ import annotations

import os
import re
from typing import List, Tuple

from tools.sandbox import resolve_safe_path

TEMPLATES = ("nextjs", "micropython", "nodejs", "python")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def scaffold_project(
    template: str,
    name: str,
    allowed_root: str,
    parent_path: str = ".",
) -> str:
    """Crea un proyecto desde plantilla embebida bajo el sandbox."""
    template = (template or "").strip().lower()
    name = (name or "").strip()

    if template not in TEMPLATES:
        return f"Plantilla no válida. Usa: {', '.join(TEMPLATES)}."

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
            os.makedirs(os.path.dirname(full_safe), exist_ok=True)
            # Directorios marcados con trailing slash en plantilla vía .gitkeep
            with open(full_safe, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            created.append(rel)
    except Exception as e:
        return f"No se pudo crear el proyecto: {e}"

    root = os.path.realpath(allowed_root)
    rel_root = os.path.relpath(dest_safe, root)
    tree = "\n".join(f"  - {p}" for p in created)
    return f"Proyecto '{template}' creado en {rel_root}:\n{tree}"


def _template_files(template: str, name: str) -> List[Tuple[str, str]]:
    if template == "python":
        return [
            (
                "README.md",
                f"# {name}\n\nProyecto Python generado por Jarvis.\n",
            ),
            (
                "main.py",
                'def main() -> None:\n    print("Hello from {name}")\n\n\nif __name__ == "__main__":\n    main()\n'.format(
                    name=name
                ),
            ),
            ("requirements.txt", ""),
            (
                ".gitignore",
                "__pycache__/\n*.py[cod]\n.venv/\nvenv/\n.env\n*.egg-info/\ndist/\nbuild/\n",
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
                    '  "scripts": {\n'
                    '    "start": "node index.js"\n'
                    "  }\n"
                    "}\n"
                ),
            ),
            (
                "index.js",
                f'console.log("Hello from {name}");\n',
            ),
            (
                "README.md",
                f"# {name}\n\nProyecto Node.js generado por Jarvis.\n",
            ),
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
            ("next.config.js", "/** @type {import('next').NextConfig} */\nconst nextConfig = {};\nmodule.exports = nextConfig;\n"),
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
                    "    <html lang=\"en\">\n"
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
                    f'  return <main><h1>{name}</h1></main>;\n'
                    "}\n"
                ),
            ),
            (
                "README.md",
                f"# {name}\n\nProyecto Next.js (App Router) generado por Jarvis.\n\n```bash\nnpm install\nnpm run dev\n```\n",
            ),
            (".gitignore", "node_modules/\n.next/\nout/\n.env*\n"),
        ]

    # micropython
    return [
        (
            "main.py",
            (
                "def main():\n"
                f'    print("Hello from {name} (MicroPython)")\n\n'
                "if __name__ == \"__main__\":\n"
                "    main()\n"
            ),
        ),
        (
            "boot.py",
            "# Runs on every boot before main.py\n",
        ),
        (
            "README.md",
            f"# {name}\n\nProyecto MicroPython generado por Jarvis.\n",
        ),
        ("lib/.gitkeep", ""),
    ]
