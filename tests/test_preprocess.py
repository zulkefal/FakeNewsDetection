from preprocess import load_data

def test_load_data_shapes():
    X, y = load_data()
    assert len(X) == len(y), "Features and labels must have the same length"
    assert len(X) > 0, "Dataset must not be empty"