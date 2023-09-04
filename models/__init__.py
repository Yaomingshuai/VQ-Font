
# generators
# from .generator import Generator
from .generator import Generator

# discriminator builder
from .discriminator import disc_builder

def generator_dispatch():
    return Generator
