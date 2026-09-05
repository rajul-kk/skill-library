"""Extraction tests for the real-model policy."""
import pytest

from agent.lm_policy import extract_program

FENCED = "Sure!\n```dsl\ngoto(forest)\ngather(wood, 2)\n```\nThat gathers wood."
PROSE_FIRST = "First we need wood.\ngoto(forest)\ngather(wood, 2)\nNow we have wood."
TRAILING = "goto(forest)\n\ngather(wood, 2)"


def test_program_is_pulled_out_of_a_code_fence():
    assert extract_program(FENCED) == "goto(forest)\ngather(wood, 2)"


def test_prose_before_the_program_is_skipped():
    assert extract_program(PROSE_FIRST) == "goto(forest)\ngather(wood, 2)"


def test_program_ends_at_the_first_blank_line():
    assert extract_program(TRAILING) == "goto(forest)"


def test_inline_comments_are_stripped():
    assert extract_program("goto(forest)  # head there") == "goto(forest)"


def test_output_with_no_dsl_at_all_is_empty():
    assert extract_program("I am not sure how to do that.") == ""


def test_unknown_operations_are_not_treated_as_dsl():
    assert extract_program("fly(moon)\ngoto(forest)") == "goto(forest)"


def test_extraction_respects_the_line_cap():
    runaway = "\n".join(["goto(forest)"] * 50)
    assert len(extract_program(runaway, max_lines=5).splitlines()) == 5


def test_extracted_programs_parse():
    from env.dsl import parse_program

    assert len(parse_program(extract_program(FENCED))) == 2
