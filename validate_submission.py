#!/usr/bin/env python3
"""
validate_submission.py
======================
Validador de archivos de submisión para MSLG-SPA 2026 (IberLEF 2026).

Uso:
    python validate_submission.py --track MSLG2SPA --submission archivo.txt [--ref referencia.txt]
    python validate_submission.py --track SPA2MSLG --submission archivo.txt [--ref referencia.txt]
    python validate_submission.py --both --mslg2spa sub1.txt --spa2mslg sub2.txt [--ref referencia.txt]

Tracks:
    MSLG2SPA  →  Entrada: glosses MSL  |  Salida esperada: texto en español
    SPA2MSLG  →  Entrada: español      |  Salida esperada: glosses MSL
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------
#  Constantes y patrones de anotacion MSL
# ---------------------------------------------

# Convenciones de anotación válidas en glosses MSL
GLOSS_ANNOTATION_PATTERNS = {
    "hyphen":    re.compile(r"^[A-ZÁÉÍÓÚÜÑ0-9]+(-[A-ZÁÉÍÓÚÜÑ0-9]+)+$"),   # YA-VEO
    "plus":      re.compile(r"^[A-ZÁÉÍÓÚÜÑ0-9]+(\+[A-ZÁÉÍÓÚÜÑ0-9]+)+$"),  # MAMÁ+PAPÁ
    "numbersign":re.compile(r"^#[A-ZÁÉÍÓÚÜÑ0-9]+$"),                        # #OK
    "dm_prefix": re.compile(r"^dm-[A-ZÁÉÍÓÚÜÑ0-9]+$"),                     # dm-LUIS
    "plain":     re.compile(r"^[A-ZÁÉÍÓÚÜÑ0-9]+$"),                        # CASA
}

# Tokens de puntuacion permitidos en ambas salidas
ALLOWED_PUNCTUATION = ".,;:!?¿¡()\"'-…“”«»"

# Patrón para detectar si una línea parece gloss (mayoría de tokens en mayúsculas)
GLOSS_LINE_RATIO_THRESHOLD = 0.5  # Si >50% tokens son gloss → línea es gloss


# ---------------------------------------------
#  Estructuras de resultado
# ---------------------------------------------

@dataclass
class ValidationError:
    level:   str   # "ERROR" | "WARNING"
    line_no: int   # 0 = error global
    message: str

@dataclass
class ValidationResult:
    track:      str
    filepath:   str
    total_lines: int = 0
    errors:     list = field(default_factory=list)
    warnings:   list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, line_no: int, msg: str):
        self.errors.append(ValidationError("ERROR", line_no, msg))

    def add_warning(self, line_no: int, msg: str):
        self.warnings.append(ValidationError("WARNING", line_no, msg))


# ---------------------------------------------
#  Funciones auxiliares
# ---------------------------------------------

def load_file(path: str) -> tuple[list[str], Optional[str]]:
    """Carga el archivo con detección de encoding. Retorna (líneas, error)."""
    p = Path(path)
    if not p.exists():
        return [], f"Archivo no encontrado: {path}"
    if p.suffix.lower() != ".txt":
        return [], f"El archivo debe tener extensión .txt, encontrado: {p.suffix}"

    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                lines = f.readlines()
            return lines, None
        except UnicodeDecodeError:
            continue
    return [], f"No se pudo decodificar el archivo. Use UTF-8."


def is_valid_gloss_token(token: str) -> bool:
    """Retorna True si el token es un gloss MSL válido según las convenciones."""
    clean = token.strip(ALLOWED_PUNCTUATION)
    if not clean:
        return True  # puntuación sola es ok
    for pattern in GLOSS_ANNOTATION_PATTERNS.values():
        if pattern.match(clean):
            return True
    return False


def looks_like_gloss_line(line: str) -> bool:
    """Heurística: ¿la línea parece una secuencia de glosses?"""
    tokens = line.strip().split()
    if not tokens:
        return False
    gloss_count = sum(1 for t in tokens if t.isupper() or is_valid_gloss_token(t))
    return (gloss_count / len(tokens)) >= GLOSS_LINE_RATIO_THRESHOLD


def check_for_control_chars(line: str, line_no: int, result: ValidationResult):
    """Detecta caracteres de control inesperados."""
    for i, ch in enumerate(line):
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ("\n", "\r", "\t"):
            result.add_error(line_no, f"Carácter de control inesperado U+{ord(ch):04X} en posición {i}.")
            break


# ---------------------------------------------
#  Validadores por track
# ---------------------------------------------

def validate_mslg2spa(lines: list[str], result: ValidationResult, ref_lines: Optional[list[str]]):
    """
    MSLG2SPA: La salida es texto en ESPAÑOL.
    - Sin tokens en mayúsculas estilo gloss (serían errores de traducción)
    - Sin anotaciones MSL (#, dm-, +)
    - Una oración por línea, no vacía
    """
    result.total_lines = len(lines)

    for i, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n\r")
        
        # Manejar formato "ID"\t"Output"
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                # El segundo elemento es el contenido real
                line_content = parts[1].strip('"')
            else:
                line_content = line.strip('"')
        else:
            line_content = line.strip('"')

        check_for_control_chars(line, i, result)

        # Linea vacia
        if not line_content.strip():
            result.add_error(i, "Linea vacia. Cada linea debe contener una traduccion.")
            continue

        # Trailing whitespace severo
        if line != line.strip() and len(line) - len(line.strip()) > 3:
            result.add_warning(i, "Espacios extra al inicio o final de la línea.")

        tokens = line.strip().split()

        # Detectar anotaciones MSL en salida española (error grave)
        for tok in tokens:
            if re.match(r"^#[A-ZÁÉÍÓÚÜÑ0-9]+$", tok):
                result.add_error(i, f"Token con '#' encontrado: '{tok}'. "
                                    "Las anotaciones MSL NO deben aparecer en la salida española.")
            if re.match(r"^dm-[A-ZÁÉÍÓÚÜÑ0-9]+", tok, re.IGNORECASE):
                result.add_error(i, f"Token con prefijo 'dm-' encontrado: '{tok}'. "
                                    "Las anotaciones MSL NO deben aparecer en la salida española.")
            if re.match(r"^[A-ZÁÉÍÓÚÜÑ]+(\+[A-ZÁÉÍÓÚÜÑ]+)+$", tok):
                result.add_error(i, f"Token con '+' encontrado: '{tok}'. "
                                    "Los signos compuestos NO deben aparecer en la salida española.")

        # Advertir si la linea parece un gloss (probablemente no traducida)
        if looks_like_gloss_line(line_content.strip()):
            result.add_warning(i, f"La linea parece ser un gloss MSL en lugar de espanol: '{line_content.strip()[:60]}...'")

        # Verificar longitud mínima razonable
        if len(tokens) < 1:
            result.add_error(i, "Traducción demasiado corta o vacía.")
        elif len(tokens) > 200:
            result.add_warning(i, f"Línea muy larga ({len(tokens)} tokens). Verifique que no haya líneas fusionadas.")

    # Verificar numero de lineas contra referencia
    if ref_lines is not None:
        # Detectar si la referencia tiene cabecera
        ref_count = len(ref_lines)
        if ref_lines and "ID" in ref_lines[0]:
            ref_count -= 1
            
        if len(lines) != ref_count:
            result.add_error(0, f"El archivo de submision tiene {len(lines)} lineas, "
                               f"pero la referencia/test tiene {ref_count} lineas (excluyendo cabecera).")


def validate_spa2mslg(lines: list[str], result: ValidationResult, ref_lines: Optional[list[str]]):
    """
    SPA2MSLG: La salida son GLOSSES MSL.
    - Tokens deben seguir las convenciones de anotación MSL
    - Generalmente en MAYÚSCULAS
    - Anotaciones especiales (#, dm-, -, +) deben usarse correctamente
    - Una secuencia de glosses por línea
    """
    result.total_lines = len(lines)

    for i, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n\r")

        # Manejar formato "ID"\t"Output"
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                line_content = parts[1].strip('"')
            else:
                line_content = line.strip('"')
        else:
            line_content = line.strip('"')

        check_for_control_chars(line, i, result)

        # Linea vacia
        if not line_content.strip():
            result.add_error(i, "Linea vacia. Cada linea debe contener una secuencia de glosses.")
            continue

        if line != line.strip() and len(line) - len(line.strip()) > 3:
            result.add_warning(i, "Espacios extra al inicio o final de la línea.")

        tokens = line.strip().split()

        invalid_tokens = []
        lowercase_tokens = []

        for tok in tokens:
            clean_tok = tok.strip("".join(ALLOWED_PUNCTUATION))
            if not clean_tok:
                continue

            # Verificar si el token es un gloss válido
            if not is_valid_gloss_token(clean_tok):
                invalid_tokens.append(tok)

            # Advertir sobre tokens en minúsculas (inusual en glosses)
            if clean_tok.islower() and len(clean_tok) > 1:
                lowercase_tokens.append(tok)

        if invalid_tokens:
            sample = invalid_tokens[:5]
            result.add_warning(i, f"Tokens posiblemente inválidos como glosses: {sample}. "
                                   "Los glosses deben estar en MAYÚSCULAS con convenciones MSL.")

        if lowercase_tokens:
            result.add_warning(i, f"Tokens en minúsculas encontrados: {lowercase_tokens[:5]}. "
                                   "Los glosses normalmente están en MAYÚSCULAS.")

        # Detectar texto en espanol sin traducir (heuristica)
        if not looks_like_gloss_line(line_content.strip()):
            result.add_warning(i, f"La linea NO parece una secuencia de glosses MSL: '{line_content.strip()[:60]}'. "
                                   "Verifique que la salida este en formato gloss.")

        # Verificar uso malformado de convenciones
        for tok in tokens:
            # Guión suelto o al inicio/final
            if tok.startswith("-") or tok.endswith("-"):
                result.add_error(i, f"Token con guión mal formado: '{tok}'. "
                                    "Formato esperado: PALABRA-PALABRA.")
            # Plus suelto
            if tok.startswith("+") or tok.endswith("+"):
                result.add_error(i, f"Token con '+' mal formado: '{tok}'. "
                                    "Formato esperado: PALABRA+PALABRA.")
            # dm- sin contenido
            if tok.lower() == "dm-":
                result.add_error(i, f"Prefijo 'dm-' sin token después: '{tok}'.")

        if len(tokens) > 150:
            result.add_warning(i, f"Secuencia muy larga ({len(tokens)} tokens). Verifique líneas fusionadas.")

    # Verificar numero de lineas contra referencia
    if ref_lines is not None:
        # Detectar si la referencia tiene cabecera
        ref_count = len(ref_lines)
        if ref_lines and "ID" in ref_lines[0]:
            ref_count -= 1
            
        if len(lines) != ref_count:
            result.add_error(0, f"El archivo de submision tiene {len(lines)} lineas, "
                               f"pero la referencia/test tiene {ref_count} lineas (excluyendo cabecera).")


# ---------------------------------------------
#  Reporte
# ---------------------------------------------

def print_report(result: ValidationResult):
    SEP = "-" * 60
    status = "VALIDO" if result.is_valid else "INVALIDO"

    print(f"\n{SEP}")
    print(f"  Track    : {result.track}")
    print(f"  Archivo  : {result.filepath}")
    print(f"  Lineas   : {result.total_lines}")
    print(f"  Estado   : {status}")
    print(f"  Errores  : {len(result.errors)}")
    print(f"  Warnings : {len(result.warnings)}")
    print(SEP)

    if result.errors:
        print("\n  [ERRORES] (deben corregirse):")
        for e in result.errors:
            loc = f"[linea {e.line_no}]" if e.line_no > 0 else "[global]"
            print(f"    {loc} {e.message}")

    if result.warnings:
        print("\n  [ADVERTENCIAS] (revisar):")
        for w in result.warnings:
            loc = f"[linea {w.line_no}]" if w.line_no > 0 else "[global]"
            print(f"    {loc} {w.message}")

    if result.is_valid and not result.warnings:
        print("\n  El archivo pasa todas las validaciones sin advertencias.")
    elif result.is_valid:
        print("\n  El archivo es valido pero tiene advertencias que conviene revisar.")

    print(SEP + "\n")


# ---------------------------------------------
#  Main
# ---------------------------------------------

def run_validation(track: str, submission_path: str, ref_path: Optional[str] = None) -> ValidationResult:
    result = ValidationResult(track=track, filepath=submission_path)

    lines, err = load_file(submission_path)
    if err:
        result.add_error(0, err)
        return result

    ref_lines = None
    if ref_path:
        ref_lines, ref_err = load_file(ref_path)
        if ref_err:
            result.add_warning(0, f"No se pudo cargar el archivo de referencia: {ref_err}")
            ref_lines = None

    if track.upper() == "MSLG2SPA":
        validate_mslg2spa(lines, result, ref_lines)
    elif track.upper() == "SPA2MSLG":
        validate_spa2mslg(lines, result, ref_lines)
    else:
        result.add_error(0, f"Track desconocido: '{track}'. Use 'MSLG2SPA' o 'SPA2MSLG'.")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validador de submisiones MSLG-SPA 2026 (IberLEF 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--track", choices=["MSLG2SPA", "SPA2MSLG"],
                      help="Track a validar.")
    mode.add_argument("--both", action="store_true",
                      help="Validar ambos tracks a la vez.")

    parser.add_argument("--submission", metavar="FILE",
                        help="Archivo de submisión (para --track).")
    parser.add_argument("--mslg2spa", metavar="FILE",
                        help="Submisión MSLG2SPA (para --both).")
    parser.add_argument("--spa2mslg", metavar="FILE",
                        help="Submisión SPA2MSLG (para --both).")
    parser.add_argument("--ref", metavar="FILE",
                        help="Archivo de referencia/test (para verificar número de líneas).")

    args = parser.parse_args()

    all_valid = True

    if args.track:
        if not args.submission:
            parser.error("--submission es requerido cuando se usa --track.")
        result = run_validation(args.track, args.submission, args.ref)
        print_report(result)
        all_valid = result.is_valid

    elif args.both:
        if not args.mslg2spa or not args.spa2mslg:
            parser.error("--mslg2spa y --spa2mslg son requeridos cuando se usa --both.")

        r1 = run_validation("MSLG2SPA", args.mslg2spa, args.ref)
        r2 = run_validation("SPA2MSLG", args.spa2mslg, args.ref)
        print_report(r1)
        print_report(r2)

        all_valid = r1.is_valid and r2.is_valid

        # Verificar consistencia entre ambas submisiones
        if r1.total_lines != r2.total_lines and r1.total_lines > 0 and r2.total_lines > 0:
            print(f"⚠️  ADVERTENCIA GLOBAL: Los dos archivos tienen distinto número de líneas "
                  f"({r1.total_lines} vs {r2.total_lines}).\n")

    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
