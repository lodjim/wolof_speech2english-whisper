import click
from rich.progress import track





@click.group(chain=False, invoke_without_command=True)
def handle_command():
    pass
 
@handle_command.command(name="description",help="The descrption of this project")
def description():
    print('The project "wolof_speech2english-whisper" is focused on developing a system that can transcribe Wolof speech into English text using a whisper model. Wolof is a language spoken in Senegal, Gambia, and Mauritania, among other West African countries.')

if __name__ == "__main__":
    handle_command()