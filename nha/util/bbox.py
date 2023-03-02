import numpy as np

class BBox:
    def __init__(self, begin, end):
        self._bbox = np.floor(np.stack(
            (begin, end),
            axis=0
        )).astype(np.int16)
    @property
    def begin(self):
        return self._bbox[0]
    @property
    def end(self):
        return self._bbox[1]
    def center(self):
        return (self.begin + self.end) / 2
    def size(self):
        return self.end - self.begin
    @staticmethod
    def bound(cls, begin, end, grid_shape):
        return np.maximum(begin, np.zeros_like(begin)), np.minimum(end, grid_shape)
    def bound(self, grid_shape):
        return BBox(*BBox.bound(*self._bbox, grid_shape))
    @staticmethod
    def shift_into_shape(begin, end, grid_shape):
        if (end - begin > grid_shape).any():
            raise RuntimeError("BBox won't fit into given shape")
        begin, end = BBox.shift(begin, end, np.maximum(-begin, 0))
        begin, end = BBox.shift(begin, end, np.minimum(grid_shape - end, 0))
        return begin, end
    @staticmethod
    def shift(begin, end, t):
        return begin + t, end + t
    
    def pad(self, padding_begin, padding_end, grid_shape):
        return BBox(*BBox.shift_into_shape(
            self.begin - padding_begin,
            self.end + padding_end,
            grid_shape
        ))
    def resize(self, new_size, grid_shape):
        return BBox(*BBox.shift_into_shape(
            self.center() - new_size // 2,
            self.center() + new_size // 2 + new_size % 2,
            grid_shape
        ))
    def __repr__(self):
        return f"BBox({self.begin.tolist()}, {self.end.tolist()})"
    
    def to_slice(self):
        return tuple([
            slice(a, b)
            for a, b in self._bbox.T
        ])