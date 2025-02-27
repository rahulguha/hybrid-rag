from util import *

from generate_data_store import *
log("info", f"************** rag started ****************")
download_corpus()
build_data_store()
log("info", f"************** Vector Store Created ****************")

