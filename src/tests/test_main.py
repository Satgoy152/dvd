import sys
import os

# Allow the test to import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the specific function we want to test
from main import parse_dynamic_kwargs

def test_parse_dynamic_kwargs_types():
    """
    Tests that the CLI parser correctly converts a raw list of strings 
    into a dictionary with the correct Python data types.
    """
    # ---------------------------------------------------------
    # 1. ARRANGE: Simulate what argparse returns for 'unknown_args'
    # ---------------------------------------------------------
    raw_cli_input = [
        "--tokenizer_max_len", "99",               # Should become an int
        "--guidance_scale", "7.5",                 # Should become a float
        "--use_fp16",                              # Flag with no value, should become True
        "--random_unused_arg", "ignore this",      # Should remain a string
        "--negative_value", "-5"                   # Testing negative ints
    ]

    # ---------------------------------------------------------
    # 2. ACT: Run the function
    # ---------------------------------------------------------
    parsed_result = parse_dynamic_kwargs(raw_cli_input)

    # ---------------------------------------------------------
    # 3. ASSERT: Verify the keys and types are exactly correct
    # ---------------------------------------------------------
    expected_result = {
        "tokenizer_max_len": 99,          
        "guidance_scale": 7.5,            
        "use_fp16": True,                 
        "random_unused_arg": "ignore this",
        "negative_value": -5              
    }
  
    assert parsed_result == expected_result
    
    assert isinstance(parsed_result["tokenizer_max_len"], int)
    assert isinstance(parsed_result["guidance_scale"], float)
    assert isinstance(parsed_result["use_fp16"], bool)
    assert isinstance(parsed_result["random_unused_arg"], str)