from inspection.jobs.job_qr import run_qr, eval_qr
from inspection.jobs.job_ocr import run_ocr, eval_ocr

JOB_REGISTRY = {
    "qr_presence": (run_qr, eval_qr),
    "ocr_text": (run_ocr, eval_ocr),
}