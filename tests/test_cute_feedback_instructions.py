"""CuTe interface instructions and one-based submission warning contract."""

from cute_rlm.evaluate import cute_turn_notice


def test_warning_starts_before_turn_28_and_reserves_final_turn_for_submission():
    assert cute_turn_notice(27, 32) is None
    warning = cute_turn_notice(28, 32)
    assert "28/32" in warning and "5 model turns remain" in warning
    assert "before the final turn" in warning
    assert "2 model turns remain" in cute_turn_notice(31, 32)
    final = cute_turn_notice(32, 32)
    assert "FINAL TURN 32/32" in final and "Submit" in final
    assert cute_turn_notice(59, 64) is None
    assert "60/64" in cute_turn_notice(60, 64)
    assert "FINAL TURN 64/64" in cute_turn_notice(64, 64)
    assert cute_turn_notice(28, None) is None
    assert "FINAL TURN 4/4" in cute_turn_notice(4, 4)
