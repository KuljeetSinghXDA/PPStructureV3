"""
PP-StructureV3 Configuration for Medical Lab Reports - CORRECTED VERSION
All models and parameters configurable through this file
Updated to use valid parameters for PaddleOCR 3.x
"""

# PP-StructureV3 Configuration for ARM64 CPU
MODELS_CONFIG = {
    # Layout Analysis Model (PP-DocLayout-L for best accuracy)
    "layout_model_dir": "PP-DocLayout-L",
    
    # OCR Models
    "det_model_dir": "PP-OCRv5_mobile_det",  # Text detection
    "rec_model_dir": "en_PP-OCRv5_mobile_rec",  # English text recognition
    
    # Processing Parameters optimized for medical lab reports
    "processing_params": {
        # Layout Analysis :cite[8]
        "layout_score_threshold": 0.5,
        "layout_nms_threshold": 0.5,
        
        # OCR Parameters :cite[10]
        "det_limit_side_len": 736,
        "det_limit_type": "min",
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.6,
        "det_db_unclip_ratio": 1.5,
        
        # Text Recognition - REMOVED INVALID drop_score parameter
        "rec_image_shape": "3, 48, 320",
        "rec_batch_num": 8,
        "max_text_length": 256,
        
        # Table Recognition :cite[4]
        "table_max_len": 488,
        "merge_no_span_structure": True,
        
        # Document Recovery
        "recovery_to_markdown": True,
        
        # Image Preprocessing
        "invert": False,
        "binarize": False,
        
        # Language
        "lang": "en",
        
        # Device - CPU for ARM64 :cite[10]
        "use_gpu": False,
        "enable_mkldnn": True,  # Optimize for CPU
        "cpu_threads": 4,
    },
    
    # PP-StructureV3 Pipeline Features :cite[8]:cite[10]
    "pipeline_features": {
        "layout": True,      # Layout analysis
        "table": True,       # Table recognition  
        "formula": True,     # Formula recognition
        "ocr": True,         # OCR text extraction
        "recovery": True,    # Document recovery to Markdown
        "kie": False,        # Key Information Extraction (disable if not needed)
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
}

def get_structurev3_kwargs():
    """
    Generate kwargs for PP-StructureV3 initialization based on configuration
    Uses valid parameters for PaddleOCR 3.x :cite[10]
    """
    config = MODELS_CONFIG
    
    # Base kwargs for pipeline initialization - ONLY USING VALID PARAMETERS
    kwargs = {
        # Model directories :cite[8]
        "layout_model_dir": config["layout_model_dir"],
        "det_model_dir": config["det_model_dir"], 
        "rec_model_dir": config["rec_model_dir"],
        
        # Device configuration :cite[10]
        "use_gpu": config["processing_params"]["use_gpu"],
        "enable_mkldnn": config["processing_params"]["enable_mkldnn"],
        "cpu_threads": config["processing_params"]["cpu_threads"],
        
        # OCR parameters - ONLY VALID ONES
        "det_limit_side_len": config["processing_params"]["det_limit_side_len"],
        "det_limit_type": config["processing_params"]["det_limit_type"],
        "det_db_thresh": config["processing_params"]["det_db_thresh"],
        "det_db_box_thresh": config["processing_params"]["det_db_box_thresh"],
        "det_db_unclip_ratio": config["processing_params"]["det_db_unclip_ratio"],
        "rec_image_shape": config["processing_params"]["rec_image_shape"],
        "rec_batch_num": config["processing_params"]["rec_batch_num"],
        "max_text_length": config["processing_params"]["max_text_length"],
        
        # Table parameters
        "table_max_len": config["processing_params"]["table_max_len"],
        "merge_no_span_structure": config["processing_params"]["merge_no_span_structure"],
        
        # Layout parameters :cite[8]
        "layout_score_threshold": config["processing_params"]["layout_score_threshold"],
        "layout_nms_threshold": config["processing_params"]["layout_nms_threshold"],
        
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
        # Enable/disable pipeline features :cite[10]
        "layout": config["pipeline_features"]["layout"],
        "table": config["pipeline_features"]["table"],
        "formula": config["pipeline_features"]["formula"], 
        "ocr": config["pipeline_features"]["ocr"],
        "recovery": config["pipeline_features"]["recovery"],
        "kie": config["pipeline_features"]["kie"],
        "use_doc_orientation_classify": config["pipeline_features"]["use_doc_orientation_classify"],
        "use_doc_unwarping": config["pipeline_features"]["use_doc_unwarping"],
        "use_textline_orientation": config["pipeline_features"]["use_textline_orientation"],
        
        # Image preprocessing
        "invert": config["processing_params"]["invert"],
        "binarize": config["processing_params"]["binarize"],
    }
    
    return kwargs
