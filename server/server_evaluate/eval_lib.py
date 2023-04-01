
from .net import Net
from .net_lib import test_model, load_data, DEVICE

# Load data
testloader, num_examples = load_data()

# Load model
model = Net().to(DEVICE)

def server_eval(model_state_dict):
    model.load_state_dict(model_state_dict)
    eval_loss, eval_accuracy = test_model(model, testloader)
    eval_results = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
    return eval_results