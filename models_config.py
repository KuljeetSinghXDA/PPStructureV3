"""
PP-StructureV3 Configuration for Medical Lab Reports
All models and parameters configurable through this file
"""

# PP-StructureV3 Configuration for ARM64 CPU
MODELS_CONFIG = {
    # Layout Analysis Model (PP-DocLayout-L for best accuracy)
    "layout_model_dir": "PP-DocLayout-L",
    
    # OCR Models
    "det_model_dir": "PP-OCRv5_mobile_det",  # Text detection
    "rec_model_dir": "en_PP-OCRv5_mobile_rec",  # English text recognition
    
    # Table Recognition (default)
    "table_model_dir": None,  # Will use default
    
    # Formula Recognition (default) 
    "formula_model_dir": None,  # Will use default
    
    # KIE Models (default)
    "kie_model_dir": None,  # Will use default
    "ser_model_dir": None,  # Will use default
    "re_model_dir": None,   # Will use default
    
    # Processing Parameters optimized for medical lab reports
    "processing_params": {
        # Layout Analysis
        "layout_score_threshold": 0.5,
        "layout_nms_threshold": 0.5,
        
        # OCR Parameters
        "det_limit_side_len": 736,
        "det_limit_type": "min",
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.6,
        "det_db_unclip_ratio": 1.5,
        
        # Text Recognition
        "rec_image_shape": "3, 48, 320",
        "rec_batch_num": 8,
        "max_text_length": 256,
        "drop_score": 0.5,
        
        # Table Recognition
        "table_max_len": 488,
        "merge_no_span_structure": True,
        
        # Document Recovery
        "recovery_to_markdown": True,
        "use_pdf2docx_api": False,
        
        # Image Preprocessing
        "invert": False,
        "binarize": False,
        "alphacolor": (255, 255, 255),
        
        # Language
        "lang": "en",
        
        # Device - CPU for ARM64
        "use_gpu": False,
        "enable_mkldnn": True,  # Optimize for CPU
        "cpu_threads": 4,
    },
    
    # PP-StructureV3 Pipeline Features
    "pipeline_features": {
        "layout": True,      # Layout analysis
        "table": True,       # Table recognition  
        "formula": True,     # Formula recognition
        "ocr": True,         # OCR text extraction
        "recovery": True,    # Document recovery to Markdown
        "kie": False,        # Key Information Extraction (disable if not needed)
    }
}

def get_structurev3_kwargs():
    """
    Generate kwargs for PP-StructureV3 initialization based on configuration
    """
    config = MODELS_CONFIG
    
    # Base kwargs for pipeline initialization
    kwargs = {
        # Model directories
        "layout_model_dir": config["layout_model_dir"],
        "det_model_dir": config["det_model_dir"], 
        "rec_model_dir": config["rec_model_dir"],
        "table_model_dir": config["table_model_dir"],
        "formula_model_dir": config["formula_model_dir"],
        "kie_model_dir": config["kie_model_dir"],
        "ser_model_dir": config["ser_model_dir"],
        "re_model_dir": config["re_model_dir"],
        
        # Device configuration
        "use_gpu": config["processing_params"]["use_gpu"],
        "enable_mkldnn": config["processing_params"]["enable_mkldnn"],
        "cpu_threads": config["processing_params"]["cpu_threads"],
        
        # OCR parameters
        "det_limit_side_len": config["processing_params"]["det_limit_side_len"],
        "det_limit_type": config["processing_params"]["det_limit_type"],
        "det_db_thresh": config["processing_params"]["det_db_thresh"],
        "det_db_box_thresh": config["processing_params"]["det_db_box_thresh"],
        "det_db_unclip_ratio": config["processing_params"]["det_db_unclip_ratio"],
        "rec_image_shape": config["processing_params"]["rec_image_shape"],
        "rec_batch_num": config["processing_params"]["rec_batch_num"],
        "max_text_length": config["processing_params"]["max_text_length"],
        "drop_score": config["processing_params"]["drop_score"],
        
        # Table parameters
        "table_max_len": config["processing_params"]["table_max_len"],
        "merge_no_span_structure": config["processing_params"]["merge_no_span_structure"],
        
        # Layout parameters
        "layout_score_threshold": config["processing_params"]["layout_score_threshold"],
        "layout_nms_threshold": config["processing_params"]["layout_nms_threshold"],
        
        # Recovery parameters
        "recovery_to_markdown": config["processing_params"]["recovery_to_markdown"],
        "use_pdf2docx_api": config["processing_params"]["use_pdf2docx_api"],
        
        # Language
        "lang": config["processing_params"]["lang"],
    }
    
    # Remove None values (use defaults)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    return kwargs

def get_pipeline_kwargs():
    """
    Generate kwargs for pipeline execution
    """
    config = MODELS_CONFIG
    
    kwargs = {
        # Enable/disable pipeline features
        "layout": config["pipeline_features"]["layout"],
        "table": config["pipeline_features"]["table"],
        "formula": config["pipeline_features"]["formula"], 
        "ocr": config["pipeline_features"]["ocr"],
        "recovery": config["pipeline_features"]["recovery"],
        "kie": config["pipeline_features"]["kie"],
        
        # Image preprocessing
        "invert": config["processing_params"]["invert"],
        "binarize": config["processing_params"]["binarize"],
        "alphacolor": config["processing_params"]["alphacolor"],
    }
    
    return kwargs
