from chess_ai.ChessDataset import ChessDataset

def test_load_samples():
    dataset = ChessDataset(100)
    assert len(dataset) > 100