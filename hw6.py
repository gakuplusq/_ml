import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data              # 數值
        self.grad = 0.0               # 對此值的梯度
        self._backward = lambda: None  # 向後傳播函數
        self._prev = set(_children)   # 前面的節點（建立計算圖用）
        self._op = _op                # 操作名稱（如 +, *, sigmoid）

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        """ e^x 的實作，導數為 e^x """
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        """ Sigmoid 函數：f(x) = 1 / (1 + e^-x)，導數為 sigmoid(x)(1 - sigmoid(x)) """
        sig = 1 / (1 + math.exp(-self.data))
        out = Value(sig, (self,), 'sigmoid')

        def _backward():
            self.grad += sig * (1 - sig) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # 拓撲排序：確保先處理前面節點
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0  # 對自己導數是 1
        for node in reversed(topo):
            node._backward()
