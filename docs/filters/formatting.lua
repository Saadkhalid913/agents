-- formatting.lua — Pandoc Lua filter for the agents report
--
-- Two transformations:
-- 1. Header: strip trailing file-path references like (path/to/file)
-- 2. CodeBlock: wrap code and diagrams in tcolorbox environments

----------------------------------------------------------------------
-- Header filter: strip trailing " (`path/to/file`)" from headings
----------------------------------------------------------------------
-- Matches pattern: ... Space "(" Code ")"
-- e.g. "Naive RAG (`1_rag/1.2_naive_rag_with_embeddings.py`)" → "Naive RAG"
function Header(el)
  local inlines = el.content
  local n = #inlines

  -- Need at least 4 trailing elements: Space, Str("("), Code, Str(")")
  if n < 4 then return nil end

  local last = inlines[n]
  local code = inlines[n - 1]
  local open = inlines[n - 2]
  local space = inlines[n - 3]

  -- Check the pattern
  if last.t == "Str" and last.text:match("^%)") and
     code.t == "Code" and
     open.t == "Str" and open.text:match("^%(") and
     space.t == "Space" then
    -- Remove the last 4 elements
    for _ = 1, 4 do
      table.remove(inlines)
    end
    -- Also trim any trailing space that may remain
    while #inlines > 0 and inlines[#inlines].t == "Space" do
      table.remove(inlines)
    end
    el.content = inlines
    return el
  end

  return nil
end

----------------------------------------------------------------------
-- CodeBlock filter: wrap in tcolorbox environments
----------------------------------------------------------------------
-- - Python/bash code → codebox + minted
-- - Diagram/text blocks → diagrambox + verbatim
-- - Plain blocks → codebox + verbatim
-- All get \needspace to avoid starting near page bottom

----------------------------------------------------------------------
-- Box alignment: fix ragged right edges in Unicode box-drawing diagrams
----------------------------------------------------------------------
-- Scans for boxes (┌...┐ / │...│ / └...┘) and aligns right edges.

-- Split a UTF-8 string into an array of single-character strings
local function utf8_chars(s)
  local chars = {}
  local i = 1
  local bytes = #s
  while i <= bytes do
    local b = s:byte(i)
    local char_len
    if b < 0x80 then char_len = 1
    elseif b < 0xE0 then char_len = 2
    elseif b < 0xF0 then char_len = 3
    else char_len = 4 end
    table.insert(chars, s:sub(i, i + char_len - 1))
    i = i + char_len
  end
  return chars
end

-- Pad or trim a line so that a box-edge character lands at target_col
local function adjust_right_edge(line, current_col, target_col)
  if current_col == target_col then return line end

  local chars = utf8_chars(line)
  local edge_char = chars[current_col]
  if not edge_char then return line end

  -- Determine fill character based on what's before the edge
  local fill = " "
  if current_col > 1 then
    local prev = chars[current_col - 1]
    if prev == "─" then fill = "─" end
  end

  -- Split into: before_edge (chars 1..current_col-1), edge, after (current_col+1..)
  local before = {}
  for c = 1, current_col - 1 do before[c] = chars[c] end
  local after = {}
  for c = current_col + 1, #chars do table.insert(after, chars[c]) end

  if target_col > current_col then
    -- Insert fill chars before edge
    local pad = target_col - current_col
    for _ = 1, pad do table.insert(before, fill) end
  else
    -- Remove chars before edge (trim padding)
    local trim = current_col - target_col
    for _ = 1, trim do
      if #before > 0 then table.remove(before) end
    end
  end

  return table.concat(before) .. edge_char .. table.concat(after)
end

local function align_boxes(text)
  local lines = {}
  for line in (text .. "\n"):gmatch("(.-)\n") do
    table.insert(lines, line)
  end

  -- Strategy: find ┌ on a line, find matching └ (same column),
  -- collect all lines between, find max right-edge column,
  -- then adjust all lines in the box to align.

  local i = 1
  while i <= #lines do
    local chars = utf8_chars(lines[i])
    local top_left_col = nil
    local top_right_col = nil

    for c = 1, #chars do
      if chars[c] == "┌" then
        top_left_col = c
      end
      if top_left_col and (chars[c] == "┐" or chars[c] == "┬") then
        top_right_col = c
        break
      end
    end

    if top_left_col and top_right_col then
      local box_start = i
      local box_end = nil
      local right_edges = {}
      right_edges[i] = top_right_col

      for j = i + 1, #lines do
        local jchars = utf8_chars(lines[j])
        if #jchars >= top_left_col then
          local ch = jchars[top_left_col]
          if ch == "└" then
            box_end = j
            for c = top_left_col + 1, #jchars do
              if jchars[c] == "┘" or jchars[c] == "┴" then
                right_edges[j] = c
                break
              end
            end
            break
          elseif ch == "│" or ch == "├" then
            -- Content line — find the rightmost │ or ┤
            for c = top_left_col + 1, #jchars do
              if jchars[c] == "│" or jchars[c] == "┤" then
                right_edges[j] = c
              end
            end
          else
            break
          end
        else
          break
        end
      end

      if box_end then
        local max_col = 0
        for _, col in pairs(right_edges) do
          if col > max_col then max_col = col end
        end
        for j = box_start, box_end do
          if right_edges[j] and right_edges[j] ~= max_col then
            lines[j] = adjust_right_edge(lines[j], right_edges[j], max_col)
          end
        end
        i = box_end + 1
      else
        i = i + 1
      end
    else
      i = i + 1
    end
  end

  -- Remove trailing empty line added by splitting
  if #lines > 0 and lines[#lines] == "" then
    table.remove(lines)
  end

  return table.concat(lines, "\n")
end

-- Heuristic: does this look like an ASCII/Unicode diagram?
local function is_diagram(text)
  -- Count lines with box-drawing characters or ASCII art patterns
  local diagram_chars = 0
  local total_lines = 0
  for line in text:gmatch("[^\n]+") do
    total_lines = total_lines + 1
    if line:match("[┌┐└┘├┤┬┴─│▼|+%-]") then
      diagram_chars = diagram_chars + 1
    end
  end
  -- If more than 30% of lines have diagram characters, it's a diagram
  return total_lines > 0 and (diagram_chars / total_lines) > 0.3
end

function CodeBlock(el)
  local lang = ""
  if el.classes and #el.classes > 0 then
    lang = el.classes[1]
  end

  local code = el.text
  -- Estimate height: ~10pt per line + 20pt box padding
  -- No cap — we want the full block on one page
  local line_count = 1
  for _ in code:gmatch("\n") do
    line_count = line_count + 1
  end
  local needspace_pt = line_count * 10 + 20

  local raw_tex

  if lang == "text" or (lang == "" and is_diagram(code)) then
    -- Diagram block → align boxes, then diagrambox + verbatim
    code = align_boxes(code)
    raw_tex = string.format(
      "\\needspace{%dpt}\n\\begin{diagrambox}\n\\begin{verbatim}\n%s\n\\end{verbatim}\n\\end{diagrambox}",
      needspace_pt, code
    )
  elseif lang == "python" or lang == "bash" or lang == "json" then
    -- Code with known language → codebox + minted
    raw_tex = string.format(
      "\\needspace{%dpt}\n\\begin{codebox}\n\\begin{minted}{%s}\n%s\n\\end{minted}\n\\end{codebox}",
      needspace_pt, lang, code
    )
  elseif lang ~= "" then
    -- Other language → codebox + minted
    raw_tex = string.format(
      "\\needspace{%dpt}\n\\begin{codebox}\n\\begin{minted}{%s}\n%s\n\\end{minted}\n\\end{codebox}",
      needspace_pt, lang, code
    )
  else
    -- No language, not a diagram → codebox + verbatim
    raw_tex = string.format(
      "\\needspace{%dpt}\n\\begin{codebox}\n\\begin{verbatim}\n%s\n\\end{verbatim}\n\\end{codebox}",
      needspace_pt, code
    )
  end

  return pandoc.RawBlock("latex", raw_tex)
end
