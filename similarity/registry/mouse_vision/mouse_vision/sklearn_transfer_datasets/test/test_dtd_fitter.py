from mouse_vision.sklearn_transfer_datasets import DTDFitter

# Model: "simplified_mousenet_single_stream"
model_name = "simplified_mousenet_single_stream"
bf = DTDFitter(model_name)
data = bf.fit("categorization", "avgpool")

