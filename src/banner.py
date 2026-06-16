import pyfiglet
from colorama import Fore, init
init(autoreset=True)

banner = pyfiglet.figlet_format("Call Me Maybe", font="slant")
spaces = "        "
banner = spaces + banner.replace('\n', f'\n{spaces}')