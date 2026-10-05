"""What each GAD view, level, table row and labelled dimension means (RCC box GADs).

Plain-language meanings used in the annotations, the training answers and the CLI. Written from the drawings
themselves; your engineers should review this file once - every answer about "what does X represent / denote"
comes from here, so a correction here corrects the whole dataset (rebuild after editing).
"""
import re

# ---------------------------------------------------------------- views (by title)
VIEWS = [
    # (title pattern, kind, what the view represents)
    (r"SECTIONAL ELEVATION.*EXISTING|SECTION\s*C\s*-\s*C|SECTION OF C\s*-\s*C", "existing_bridge_section",
     "a section through the EXISTING bridge next to the proposed one. It shows the existing opening and its levels "
     "(existing rail, formation, soffit, bed and founding levels), so the new box can be compared with it."),
    (r"SECTIONAL ELEVATION|LONGITUDINAL SECTION", "sectional_elevation",
     "a vertical section cut along the length of the box barrel (along the direction of flow, across the tracks). "
     "It shows the embankment with the existing and proposed tracks on top and their centre lines and track-centre "
     "distances, the box underneath with its barrel length, the rail, formation, top-of-box, soffit, bed and "
     "founding levels, the HFL, and the works at the ends of the barrel (face/return walls, drop and toe walls, "
     "stone pitching and other protection)."),
    (r"HALF BOTTOM PLAN|HALF TOP PLAN", "half_plan",
     "the plan (view from above) of the box, drawn in two halves on either side of its centre line: one half shows "
     "the top (top slab with the tracks over it), the other half the bottom (base slab and foundation outline). "
     "It shows the barrel length, the face and return walls, the protection works, the section lines A-A and B-B "
     "and the direction of flow."),
    (r"SECTION\s*B\s*-\s*B", "section_bb",
     "the section through the box structure along the cut line B-B marked on the plan. It shows the thicknesses "
     "and levels of the part it cuts (slabs, walls, base slab key, levelling and wearing courses) and the ground "
     "treatment below it."),
    (r"KEY PLAN", "key_plan",
     "the location plan: where the bridge sits relative to the existing UP and DN lines and the proposed 3rd line, "
     "its chainage, the stations on either side (direction to ...), the railway boundary and the curve data of the "
     "alignment. It is a sketch for location, not a dimensioned drawing."),
    (r"BORE\s*LOG|TRAIL\s*PIT|TRIAL\s*PIT", "bore_log",
     "the soil profile from the bore hole (or trial pit) at the chainage in its title: the soil layers with their "
     "reduced levels (RL) and depths, and the safe bearing capacity (SBC) at each depth. It shows what the box "
     "foundation will sit on."),
    (r"L\s*-\s*SECTION OF ROAD", "road_lsection",
     "the longitudinal section of the road passing through the subway: existing road levels and offsets along the "
     "road, the slope, and the proposed road level at the box."),
    (r"L\s*-\s*SECTION OF DRAIN", "drain_lsection",
     "the longitudinal section of the drain along the subway: invert levels and offsets of the drain."),
    (r"CROSS DRAIN", "cross_drain_section", "a section through the cross drain at the location marked X."),
    (r"TYPICAL DETAILS? OF|DETAILS OF .*WALL", "wall_detail",
     "a typical cross-section of the wall named in the title (face, toe, drop, curtain or return wall): its "
     "thicknesses, height, foundation and levelling course, and its top and founding levels."),
    (r"SECTION OF ABUTMENT", "abutment_section", "a section through the abutment with its dimensions and levels."),
    (r"DETAIL", "detail",
     "an enlarged detail of the part marked with the same letter on the other views, drawn at a larger scale so "
     "its small dimensions can be shown."),
]


def view_kind(title):
    t = re.sub(r"\s+", " ", title.upper())
    for pat, kind, what in VIEWS:
        if re.search(pat, t):
            return kind, what
    return "other", "a view of the structure (see its title)."


# ---------------------------------------------------------------- levels (labelled values in m)
LEVELS = [
    # (label pattern, kind, meaning) - first match wins, so the specific ones come first
    (r"\bSCOUR\b", "scour_level", "the lowest level the stream bed is expected to scour down to in a flood"),
    (r"\bRAIL\b", "rail_level", "the level of the top of the rail"),
    (r"FORMATION|\bF\.?L\.?\b(?!.*WALL)", "formation_level",
     "the level of the top of the embankment (the formation) on which the ballast and track rest"),
    (r"TOP OF (THE )?(RCC )?(BOX|SLAB)|BOX TOP|TOP SLAB", "top_of_box_level", "the level of the top surface of the box's top slab"),
    (r"SOFFIT", "soffit_level", "the level of the underside of the top slab, i.e. the top of the clear opening"),
    (r"\bO\.?H\.?F\.?L", "observed_hfl", "the observed high flood level: the highest flood level recorded at the site"),
    (r"\bC\.?H\.?F\.?L", "calculated_hfl", "the calculated high flood level from the hydraulic design"),
    (r"\bH\.?F\.?L|HIGH FLOOD", "hfl", "the high flood level: the highest water level the bridge is designed for"),
    (r"\bL\.?W\.?L", "lwl", "the low water level"),
    (r"INVERT|\bI\.?L\.?\b", "invert_level",
     "the level of the inside bottom of the opening or drain (the top of the bottom slab where water flows)"),
    (r"TOP OF .*WALL|WALL TOP|TOP OF COPING|COPING", "wall_top_level", "the level of the top of the wall named in the label"),
    (r"FOUND|\bFDN|\bFND", "founding_level",
     "the level of the bottom of the foundation (the underside of the base slab or wall footing, on the levelling course)"),
    (r"\bBED\b", "bed_level", "the level of the stream bed at the bridge"),
    (r"\bROAD\b", "road_level", "the level of the road surface"),
    (r"GROUND|\bN\.?G\.?L|\bG\.?L\.?\b", "ground_level", "the natural ground level"),
    (r"\bR\.?L\.?\b|\bLVL\b|LEVEL", "reduced_level", "a reduced level: the height of that point above the survey datum"),
]
OF_ELEMENT = re.compile(r"\bOF\s+((?:RCC\s+)?BOX|R/?\s?WALL|RETURN WALL|FACE WALL|TOE WALL|DROP WALL|CURTAIN WALL|WING WALL|"
                        r"RETAINING WALL|SLAB|ABUTMENT|DRAIN)\b", re.I)


def level_kind(label):
    t = re.sub(r"\s+", " ", label.upper())
    for pat, kind, meaning in LEVELS:
        if re.search(pat, t):
            return kind, meaning
    return "reduced_level", LEVELS[-1][2]


def status_of(label):
    t = label.upper()
    if re.search(r"\bPROP(OSED)?\b\.?", t):
        return "proposed"
    if re.search(r"\bEX(IST(ING)?)?\b\.?|\bEXG\b", t):
        return "existing"
    return None


# ---------------------------------------------------------------- table rows
TABLE_ROWS = {
    "BRIDGE NO": ("bridge_no", "the bridge number"),
    "SPAN": ("span", "the opening: number of cells x clear width x clear height (m)"),
    "TYPE OF SUPERSTRUCTURE": ("superstructure", "the type of superstructure (the part carrying the track)"),
    "TYPE OF SUBSTRUCTURE": ("substructure", "the type of substructure (the supports)"),
    "TYPE OF FOUNDATION": ("foundation", "the type of foundation"),
    "CATCHMENT AREA": ("catchment_area", "the area of land draining to the bridge"),
    "WATER WAY REQUIRED": ("waterway_required", "the waterway (flow area) the opening must provide"),
    "WATERWAY REQUIRED": ("waterway_required", "the waterway (flow area) the opening must provide"),
    "WATER WAY PROVIDED": ("waterway_provided", "the waterway (flow area) the opening provides"),
    "OHFL": ("observed_hfl", "the observed high flood level"),
    "CHFL": ("calculated_hfl", "the calculated high flood level"),
    "HFL": ("hfl", "the high flood level"),
    "BED LEVEL": ("bed_level", "the stream bed level"),
    "VERTICAL CLEARANCE": ("vertical_clearance", "the clear height between the HFL and the underside of the top slab (soffit)"),
    "FREE BOARD": ("free_board", "the height from the HFL to the formation level"),
    "MAX. SCOUR DEPTH": ("max_scour_depth", "the maximum depth of scour below the bed in a flood"),
    "MAX. SCOUR LEVEL": ("max_scour_level", "the lowest level the bed may scour down to"),
    "HEIGHT OF WATER": ("height_of_water", "the depth of water at HFL above the bed"),
    "VELOCITY": ("velocity", "the flow velocity through the opening at design discharge"),
    "DESIGN DISCHARGE": ("design_discharge", "the flood discharge the opening is designed for"),
    "SEISMIC ZONE": ("seismic_zone", "the earthquake zone of the site"),
    "RAIL LEVEL": ("rail_level", "the level of the top of the rail"),
    "FORMATION LEVEL": ("formation_level", "the level of the top of the embankment"),
    "ALIGNMENT": ("alignment", "whether the track is straight or on a curve at the bridge"),
    "GRADE": ("grade", "the track gradient at the bridge (R = rise, F = fall, 1 in n)"),
    "ROAD LEVEL": ("road_level", "the road surface level"),
    "LOADING": ("loading", "the loading standard"),
}


def table_row_kind(desc):
    d = re.sub(r"[\s.]+", " ", desc.upper()).strip()
    best = None
    for key, val in TABLE_ROWS.items():
        k = re.sub(r"[\s.]+", " ", key).strip()
        if d.startswith(k) or k in d:
            if best is None or len(k) > len(best[0]):
                best = (k, val)
    return best[1] if best else (re.sub(r"\W+", "_", d.lower()).strip("_"), None)


# ---------------------------------------------------------------- labelled dimensions (mm unless stated)
LABELLED_DIMS = [
    # (pattern with the value in group "v", kind, meaning template; {what} = the rest of the label)
    (r"^T\s*/\s*C\s*[=:]?\s*(?P<v>\d+(?:\.\d+)?)", "track_centres",
     "the track-centre distance: the distance between the centre lines of {between}"),
    (r"BARREL\s+LENGTH\s*[=:]?\s*(?P<v>\d+(?:\.\d+)?)", "barrel_length",
     "the barrel length: the length of the box measured along its barrel (across the tracks)"),
    (r"^(?P<v>\d+(?:\.\d+)?)\s*(?:MM|mm)?\s*THK\.?\s*(?P<what>.+)", "thickness", "the thickness of the {what}"),
    (r"^(?P<what>.+?)\s+(?P<v>\d+(?:\.\d+)?)\s*(?:MM|mm)?\s*THK\b", "thickness", "the thickness of the {what}"),
    (r"(?P<v>\d+(?:\.\d+)?)\s*(?:MM|mm)\s*GAP\s*(?P<what>.*)", "gap", "a gap of this width between the new box and the existing structure, filled as described"),
    (r"[Ø∅]\s*(?P<v>\d+(?:\.\d+)?)\s*(?:mm|MM)?\s*(?P<what>.*)", "diameter", "the diameter of the {what}"),
]
SLOPE_RE = re.compile(r"^(?P<a>\d+(?:\.\d+)?)\s*:\s*(?P<b>\d+(?:\.\d+)?)$")
GRADIENT_RE = re.compile(r"\b1\s*IN\s*(?P<n>\d+(?:\.\d+)?)", re.I)

ABBREVIATION_HINTS = {
    "T/C": "track centres (distance between the centre lines of two tracks)",
    "SBC": "safe bearing capacity of the soil",
    "THK": "thick (thickness)",
    "C/C": "centre to centre (spacing)",
    "TYP": "typical",
    "LVL": "level",
    "FDN": "foundation",
    "U/S": "upstream side",
    "D/S": "downstream side",
    "R/WALL": "return wall",
    "PCC": "plain cement concrete",
    "RCC": "reinforced cement concrete",
    "RUB": "road under bridge (a box carrying the railway over a road)",
    "ROW": "railway boundary (right of way)",
}
