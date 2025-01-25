import pytest
from plot import parse_training_line

def test_parse_training_line():
    # Test input line
    line = '[ 2024-12-14 11:01:36 ] Epoch [0] Step [11,134 / 21,977] Batch [111,349 / 219,770] Lr: [0.001], Avg Loss [0.660], Rank Corr.: [17.412%], Examples: 1,920   1,253.740 ms,    15,000.29 s total'
    
    # Parse the line
    result = parse_training_line(line)
    
    # Expected output
    expected = {
        'timestamp': '2024-12-14 11:01:36',
        'epoch': '0',
        'step': '11134',
        'total_steps': '21977',
        'batch': '111349',
        'total_batches': '219770',
        'learning_rate': '0.001',
        'avg_loss': '0.660',
        'rank_correlation': '17.412',
        'examples': '1920',
        'time_ms': '1253.740',
        'total_time_s': '15000.29'
    }
    
    # Verify each field matches expected value
    for key, expected_value in expected.items():
        assert result[key] == expected_value, f"Mismatch in {key}: expected {expected_value}, got {result[key]}"

def test_parse_training_line_with_leading_zeros():
    # Test input line with leading zeros in total_time_s
    line = '[ 2024-12-14 11:01:36 ] Epoch [1] Step [11,134 / 21,977] Batch [111,349 / 219,770] Lr: [0.001], Avg Loss [0.660], Rank Corr.: [17.412%], Examples: 1,920   1,253.740 ms,    091.325 s total'
    
    result = parse_training_line(line)
    
    # Should be 1091.325 (91.325 + 1000 for epoch 1)
    assert result['total_time_s'] == '1091.325', f"Expected 1091.325, got {result['total_time_s']}" 

def test_parse_training_line_with_comma_times():
    # Test input line with comma in time values
    line = '[ 2024-12-14 11:01:36 ] Epoch [1] Step [11,134 / 21,977] Batch [111,349 / 219,770] Lr: [0.001], Avg Loss [0.660], Rank Corr.: [17.412%], Examples: 1,920   1,253.740 ms,    1,091.325 s total'
    
    result = parse_training_line(line)
    
    assert result['time_ms'] == '1253.740', f"Expected 1253.740, got {result['time_ms']}"
    assert result['total_time_s'] == '2091.325', f"Expected 2091.325, got {result['total_time_s']}"  # 1091.325 + 1000 for epoch 1 

def test_extract_training_lines():
    # Create a temporary file with mixed content
    import tempfile
    
    content = '''[ 2024-12-14 15:05:08 ] Testing epoch 0                                                             90.650 ms,    29,612.35 s total
[ 2024-12-14 15:16:16 ] Epoch [0] Evaluation, Avg Loss [8.110], Rank Corr. [17.354%]           668,267.604 ms,    30,280.62 s total
[ 2024-12-14 15:16:16 ] Training epoch 1                                                            90.204 ms,    30,280.71 s total
[ 2024-12-14 15:16:16 ] Epoch [1] Step [11,134 / 21,977] Batch [111,349 / 219,770] Lr: [0.001], Avg Loss [0.660], Rank Corr.: [17.412%], Examples: 1,920   1,253.740 ms,    15,000.29 s total'''
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(content)
        temp_file = f.name
    
    try:
        # Test the function
        from plot import extract_training_lines
        lines = extract_training_lines(temp_file)
        
        # Should only match the training line
        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"
        assert "Step" in lines[0] and "Batch" in lines[0], "Matched line should contain Step and Batch info"
        
    finally:
        # Clean up
        import os
        os.unlink(temp_file) 