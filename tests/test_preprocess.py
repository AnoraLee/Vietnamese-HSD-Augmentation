from src.utils.preprocess import get_text_preprocessor, TextPreprocessor

def test_get_text_preprocessor_returns_instance():
    preprocessor = get_text_preprocessor()
    assert isinstance(preprocessor, TextPreprocessor)


def test_clean_text_returns_string():
    preprocessor = get_text_preprocessor()
    raw_text = "HỌC KỲ ko ổn VL"
    cleaned = preprocessor.clean_text(raw_text)
    
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0