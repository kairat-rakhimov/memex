from memex.kg_query.grounding import locate_quote


def test_locate_quote_exact_match() -> None:
    chunk = "The catalyst showed 491 mV overpotential at 10 mA/cm2."
    span = locate_quote("491 mV overpotential", chunk)
    assert span is not None
    s, e = span
    assert chunk[s:e] == "491 mV overpotential"


def test_locate_quote_whitespace_normalized_match() -> None:
    # PDF prose: a newline + a double space the stored quote won't reproduce.
    chunk = "The  catalyst\nshowed 491 mV overpotential."
    span = locate_quote("catalyst showed 491 mV", chunk)
    assert span is not None
    s, e = span
    # the original span carries the original whitespace…
    assert chunk[s:e] == "catalyst\nshowed 491 mV"
    # …and collapses back to exactly the quote.
    assert " ".join(chunk[s:e].split()) == "catalyst showed 491 mV"


def test_locate_quote_none_when_absent_or_empty() -> None:
    chunk = "Some unrelated prose about hydrogen."
    assert locate_quote("not present in the chunk", chunk) is None
    assert locate_quote("", chunk) is None
    assert locate_quote("   ", chunk) is None
    assert locate_quote("anything", "") is None
