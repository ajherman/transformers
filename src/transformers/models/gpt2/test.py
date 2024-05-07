import sys

# Path to the 'src' directory of your local transformers repository
path_to_transformers = '/home/ari/transformers/src'

# Prepend this path to sys.path
if path_to_transformers not in sys.path:
    sys.path.insert(0, path_to_transformers)

# # Now import the transformers
# from transformers import GPT2Model, GPT2Config

# # Use the module as usual
# config = GPT2Config.from_pretrained('gpt2')
# model = GPT2Model(config)

# print("Using transformers from:", GPT2Model.__module__)

from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

# Load pre-existing config
config = GPT2Config.from_pretrained('gpt2')

# Print all attributes and their values
for attr_name, attr_value in vars(config).items():
    print(attr_name)

if hasattr(config, 'hidden_size'):
    print("hidden_size:", getattr(config, 'hidden_size'))
else:
    print("hidden_size is not an attribute of GPT2Config."

# Create model from the loaded configuration
model = GPT2Model(config)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
