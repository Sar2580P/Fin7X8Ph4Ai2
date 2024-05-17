from torchview import draw_graph

class Utils :
    def __init__(self):
        pass
    
    @classmethod
    def callback_points():
        callback_points = [
            # Run in trainer
            "on_pretrain_routine_start", "on_pretrain_routine_end","on_train_start","on_train_epoch_start","on_train_batch_start",
            "optimizer_step", "on_before_zero_grad","on_train_batch_end","on_train_epoch_end","on_model_save","on_train_end","on_params_update",
            "teardown", "on_fit_epoch_end",  # fit = train + val
            
            # Run in validator
            "on_val_start","on_val_batch_start", "on_val_batch_end","on_val_end",
            
            # Run in predictor
            "on_predict_start","on_predict_batch_start", "on_predict_postprocess_end","on_predict_batch_end","on_predict_end",
            
            # Run in exporter
            "on_export_start","on_export_end",
            
            ] 
        x = '\n- '.join(callback_points)
        print(f"Available callback points are: {x}")
    
    @classmethod    
    def models():
        from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
        x = '\n- '.join(GITHUB_ASSETS_STEMS)
        print(f"Available models are: {x}")
    
    @classmethod    
    def plot_model(config , model):
        model_graph = draw_graph(model, input_size=(config['BATCH_SIZE'] , config['C'] , config['H'] , config['W']), graph_dir ='TB', expand_nested=True,
                                    graph_name=config['model_name'],save_graph=True,filename=config['model_name'], 
                                    directory=config['dir'], depth = 4)
        model_graph.visual_graph



