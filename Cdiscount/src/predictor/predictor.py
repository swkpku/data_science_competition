import time
import torch
import csv
        
class Predictor():
    def __init__(self, test_dataloader, model, config):
        self.test_dataloader = test_dataloader
        self.model = model
        self.config = config
        
        # cut the classifier layer
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        
    def run(self):
        torch.backends.cudnn.benchmark = True # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                                              # If this is set to false, uses some in-built heuristics that might not always be fastest.
        
        # switch to evaluate mode
        self.model.eval()

        # prediction result
        prod_ids = []
        predictions = []
        
        # prediction
        print("start prediction")
        end = time.time()
        for i, (img, _, prod_id) in enumerate(self.test_dataloader):
            # measure data loading time
            data_time = time.time() - end
            input_var = torch.autograd.Variable(img)

            # compute output
            output = self.model(input_var)
            _, predicted = torch.max(output.data, 1)

            prod_ids.append(prod_id)
            predictions.append(predicted)
            
            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            if i % self.config['print_freq'] == 0:
                print('Iter: [{0}/{1}]\t'
                      'Time {batch_time:.3f}\t'
                      'Data {data_time:.3f}\t'.format(
                       i, len(self.test_dataloader), batch_time=batch_time,
                       data_time=data_time))

        with open(self.config['pred_filename'], "w") as outfile:
            csvwriter = csv.writer(outfile)
            csvwriter.writerows(zip(prod_ids, predictions))   

    
def get_predictor(test_dataloader, model, config):
    return Predictor(test_dataloader, model, config)