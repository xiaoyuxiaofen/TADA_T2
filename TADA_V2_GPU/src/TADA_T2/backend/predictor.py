'''
GPU-accelerated predictor.

Data flow is now entirely on GPU as TF Tensors:
    sequences → create_features (GPU) → scale_features (GPU) → model.predict (GPU)

No NumPy intermediates. No CPU↔GPU transfers until final result.
'''
import importlib.resources
from tensorflow import convert_to_tensor

from TADA_T2.backend.features import create_features, scale_features
from TADA_T2.backend.model import TadaModel


def get_model_path():
    model_weights_path = importlib.resources.files('TADA_T2.data') / 'tada.14-0.02.hdf5'
    return str(model_weights_path)


# Cache model across calls
_model_cache = None


def predict_tada(sequences, return_both_values=False):
    '''
    Predict TADA scores — fully GPU pipeline.

    Parameters
    ----------
    sequences : list of str
        List of 40-amino-acid sequences.
    return_both_values : bool
        If True, return both softmax outputs. Default False (TAD score only).

    Returns
    -------
    list of float (or list of [float, float] if return_both_values=True)
    '''
    global _model_cache

    if not isinstance(sequences, list):
        raise Exception('Sequences must be input as a list!')

    SEQUENCE_WINDOW = 5
    STEPS = 1
    LENGTH = 40

    # Step 1: Feature computation — stays on GPU as TF Tensor
    features = create_features(sequences, SEQUENCE_WINDOW, STEPS, LENGTH)

    # Step 2: Feature scaling — stays on GPU, vectorized TF ops
    features = scale_features(features)

    # Step 3: Load model (cached)
    if _model_cache is None:
        _model_cache = TadaModel().create_model()
        _model_cache.load_weights(str(get_model_path()))

    # Step 4: Predict — model.predict handles GPU placement automatically
    # features is already a TF tensor on the correct device
    predictions = _model_cache.predict(features, verbose=0)

    # Step 5: Extract results (move to CPU only at the very end)
    if return_both_values:
        return predictions.tolist()
    return [float(pred[0]) for pred in predictions]
