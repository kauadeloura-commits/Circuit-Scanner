"""Model-aware cache identity for repeated image views."""

def prediction_key(model,image,threshold,size):
    import hashlib
    return (id(model),image.shape,image.dtype.str,hashlib.sha256(image.tobytes()).hexdigest(),float(threshold),int(size))
