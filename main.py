import click
from rich.progress import track
from librairies.utils import WhisperFinetuner

@click.group(chain=False, invoke_without_command=True)
def handle_command():
    pass
 
@handle_command.command(name="description",help="The descrption of this project")
def description():
    print('The project "wolof_speech2english-whisper" is focused on developing a system that can transcribe Wolof speech into English text using a whisper model. Wolof is a language spoken in Senegal, Gambia, and Mauritania, among other West African countries.')

@handle_command.command(name="train",help="train the model")
@click.option('--model_base',help='The model that you want to use example: whisper-small')
@click.option('--path2dataset',help='give the path to your json dataset')
@click.option('--output_dir',help="this is the path to directory where you save your model")
@click.option('--per_device_train_batch_size',help="give your batch size")
@click.option('--lr',help="learning rate")
def train(model_base,path2dataset,output_dir,per_device_train_batch_size,lr):
    whisper_finetuner = WhisperFinetuner(model_base=model_base,path2dataset=path2dataset)
    whisper_finetuner.train_model(output_dir=output_dir,per_device_train_batch_size=per_device_train_batch_size,lr=lr)
if __name__ == "__main__":
    handle_command()