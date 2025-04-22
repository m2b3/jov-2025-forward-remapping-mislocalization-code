import os
import subprocess
import shutil

TASK_DIAGRAM_BASE   = "../img/task_diagram"
def compile_task_diagram(saccade_amplitude) -> bool:
    """
    Returns whether required binaries were available
    """
    print(f"Current working directory: {os.getcwd()}")

    if shutil.which("typst") is None or shutil.which("gs") is None:
        return False

    typ_file = f"{TASK_DIAGRAM_BASE}.typ"
    pdf_file = f"{TASK_DIAGRAM_BASE}.pdf"
    png_file = f"{TASK_DIAGRAM_BASE}.png"

    if not os.path.exists(pdf_file) or os.path.getmtime(typ_file) > os.path.getmtime(pdf_file):
        argv = ["typst", "compile", typ_file, "-f", "pdf", "--input", f"saccade_amplitude={saccade_amplitude}"]
        print(f"Running: {' '.join(argv)}")
        subprocess.run(argv)

    if not os.path.exists(png_file) or os.path.getmtime(pdf_file) > os.path.getmtime(png_file):
        gs_argv = ["gs", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pngalpha", "-r1000", f"-sOutputFile={png_file}", pdf_file]
        print(f"Running: {' '.join(gs_argv)}")
        subprocess.run(gs_argv)

    return True
