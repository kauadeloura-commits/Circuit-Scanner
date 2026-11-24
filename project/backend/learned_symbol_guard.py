"""A missing template match is uncertainty, not proof that a learned symbol is text."""

_PHOTO_SOURCES = {'original_color','normalized_grayscale','normalized_grayscale_high_resolution',
                  'original_color_high_resolution',
                  'secondary_model_original',
                  'whole_circuit_context','component_ink_normalized','paper_context_padding',
                  'color_invariant_ink'}

def preservation_evidence(detection, geometry):
    if geometry.get('hard_topology_rejection') or geometry.get('junction_rejected'):
        return None
    reason = str(geometry.get('reason',''))
    if reason in {'aligned_glyphs_without_component_leads', 'isolated_text_run'} or reason.startswith('supervised_text_'):
        return None
    # These failures test an ideal plate/lead template. They are not independent
    # text evidence: broken pen strokes and standard outlined plates also fail it.
    template_failures = {'plate_strokes_without_opposed_connected_capacitor_leads',
                         'connected_glyph_strokes_instead_of_separate_capacitor_plates',
                         'parallel_glyphs_without_two_battery_leads',
                         'round_glyph_without_two_source_leads'}
    if geometry.get('hard_text_rejection') and reason not in template_failures:
        return None
    sources = set(detection.get('inference_sources',()))
    if not sources.intersection(_PHOTO_SOURCES):
        return None
    confidence = float(detection.get('best_learned_confidence', detection.get('best_view_confidence', detection.get('confidence',0))))
    learned_sources=[s for s in sources if not s.endswith('_geometry') and s!='manual_review']
    fragmented = detection.get('class') in {'resistor','gnd'} and len(learned_sources)>=3
    minimum = .60 if fragmented else .85
    if confidence < minimum or len(learned_sources)<2:
        return None
    return dict(method='strong_learned_photo_proposal_without_contradicting_text_or_junction',
                confidence=confidence,photo_sources=sorted(sources.intersection(_PHOTO_SOURCES)),
                geometry_is_inconclusive=True)
