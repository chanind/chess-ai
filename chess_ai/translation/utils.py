from typing import Tuple
import chess


def get_coords(square: int) -> Tuple[int, int]:
    x_coord = square % 8
    y_coord = square // 8
    return (x_coord, y_coord)


def transform_board_index(index: int, color: chess.Color):
    if color == chess.WHITE:
        return index
    return 63 - index
