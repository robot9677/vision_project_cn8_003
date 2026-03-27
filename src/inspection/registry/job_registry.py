from inspection.jobs.job_qr import run_qr, eval_qr
from inspection.jobs.job_ocr import run_ocr, eval_ocr
from inspection.jobs.job_barcode import run_barcode, eval_barcode

JOB_REGISTRY = {
    "qr_presence": (run_qr, eval_qr),
    "ocr_text": (run_ocr, eval_ocr),
    "barcode_1d": (run_barcode, eval_barcode),
}