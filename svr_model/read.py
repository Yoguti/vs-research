import struct

with open('svr_params.bin', 'rb') as f:
    n_vectors = struct.unpack('I', f.read(4))[0]
    dim = struct.unpack('I', f.read(4))[0]

    support_vectors = []
    for _ in range(n_vectors):
        vec = struct.unpack('f' * dim, f.read(4 * dim))
        support_vectors.append(vec)

    dual_coef = struct.unpack('f' * n_vectors, f.read(4 * n_vectors))

    intercept = struct.unpack('f', f.read(4))[0]

print(support_vectors, dual_coef, intercept)
