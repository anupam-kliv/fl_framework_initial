from server_lib import load_data, get_net, test_model

def server_eval(model_state_dict, config):
    device = config['device']
    testloader, _ = load_data(config)
    model = get_net(config)
    model = model.to(device)
    model.load_state_dict(model_state_dict)
    
    eval_loss, eval_accuracy = test_model(model, testloader)
    eval_results = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
    return eval_results