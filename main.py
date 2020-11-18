import cv2

from tabdetactor.parsers.lattice import Lattice

from tabdetactor.image_processing import adaptive_threshold, find_lines, find_contours, find_joints

from tabdetactor.utils import segments_in_bbox, merge_close_lines, get_table_index, compute_accuracy, compute_whitespace

from tabdetactor.core import Table
import os

import pandas as pd

process_background=False
threshold_blocksize=15
threshold_constant=-2

table_regions = None
table_areas = None
line_scale = 15
copy_text = None
shift_text = ["l", "t"]
split_text = False
flag_size = False
strip_text = ""
line_tol = 2
joint_tol = 2
iterations = 0
resolution = 300


img, threshold = adaptive_threshold(r'C:\DEV\Table_Detection\data\fact230001-1.png',
            process_background=process_background,
            blocksize=threshold_blocksize,
            c=threshold_constant)



regions = None

vertical_mask, vertical_segments = find_lines(
                threshold,
                regions=regions,
                direction="vertical",
                line_scale=line_scale,
                iterations=iterations,
            )



horizontal_mask, horizontal_segments = find_lines(
                threshold,
                regions=regions,
                direction="horizontal",
                line_scale=line_scale,
                iterations=iterations,
            )

contours = find_contours(vertical_mask, horizontal_mask)


for i, c1 in enumerate(contours):
    cv2.rectangle(img, (int(c1[0]), int(c1[1])), (int(c1[0]+c1[2]), int(c1[1])), (0, 255, 255), 3)

img = cv2.resize(img,(800,700))
cv2.imshow("with_line", img)
cv2.waitKey(0)
cv2.destroyWindow("with_line") #close the window

table_bbox = find_joints(contours, vertical_mask, horizontal_mask)


def _generate_columns_and_rows(table_idx, tk):
    # select elements which lie within table_bbox

    v_s, h_s = segments_in_bbox(
        tk, vertical_segments, horizontal_segments
    )
    cols, rows = zip(*table_bbox[tk])
    cols, rows = list(cols), list(rows)
    cols.extend([tk[0], tk[2]])
    rows.extend([tk[1], tk[3]])
    #sort horizontal and vertical segments
    cols = merge_close_lines(sorted(cols), line_tol=line_tol)
    rows = merge_close_lines(sorted(rows, reverse=True), line_tol=line_tol)
    # make grid using x and y coord of shortlisted rows and cols
    cols = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
    rows = [(rows[i], rows[i + 1]) for i in range(0, len(rows) - 1)]

    return cols, rows, v_s, h_s

def _generate_table(self, table_idx, cols, rows, **kwargs):
    v_s = kwargs.get("v_s")
    h_s = kwargs.get("h_s")
    if v_s is None or h_s is None:
        raise ValueError("No segments found on {}".format(self.rootname))

    table = Table(cols, rows)
    # set table edges to True using ver+hor lines
    table = table.set_edges(v_s, h_s, joint_tol=joint_tol)
    # set table border edges to True
    table = table.set_border()
    # set spanning cells to True
    table = table.set_span()

    pos_errors = []
    # TODO: have a single list in place of two directional ones?
    # sorted on x-coordinate based on reading order i.e. LTR or RTL
    for direction in ["vertical", "horizontal"]:
        for t in self.t_bbox[direction]:
            indices, error = get_table_index(
                table,
                t,
                direction,
                split_text=self.split_text,
                flag_size=self.flag_size,
                strip_text=self.strip_text,
            )
            if indices[:2] != (-1, -1):
                pos_errors.append(error)
                indices = Lattice._reduce_index(
                    table, indices, shift_text=self.shift_text
                )
                for r_idx, c_idx, text in indices:
                    table.cells[r_idx][c_idx].text = text
    accuracy = compute_accuracy([[100, pos_errors]])

    if copy_text is not None:
        table = Lattice._copy_spanning_text(table, copy_text=self.copy_text)

    data = table.data
    table.df = pd.DataFrame(data)
    table.shape = table.df.shape

    whitespace = compute_whitespace(data)
    table.flavor = "lattice"
    table.accuracy = accuracy
    table.whitespace = whitespace
    table.order = table_idx + 1
    table.page = int(os.path.basename(self.rootname).replace("page-", ""))

    # for plotting
    _text = []
    _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.horizontal_text])
    _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.vertical_text])
    table._text = _text
    table._segments = (self.vertical_segments, self.horizontal_segments)
    table._textedges = None

    return table

_tables = []
# sort tables based on y-coord
print(table_bbox)
for table_idx, tk in enumerate(
    sorted(table_bbox.keys(), key=lambda x: x[1], reverse=True)
):
    cols, rows, v_s, h_s = _generate_columns_and_rows(table_idx, tk)


    table = Table(cols, rows)
    print(table.cells)

    #table = _generate_table(table_idx, cols, rows, v_s=v_s, h_s=h_s)
    #table._bbox = tk
    #_tables.append(table)

    cv2.imshow("with_line", cv2.resize(mask, (800, 700)))
    cv2.waitKey(0)
    cv2.destroyWindow("with_line")  # close the window

    for i, c1 in enumerate(table.cells):
        for c2 in c1:
            cv2.rectangle(img, (int(c2.x1), int(c2.y1)), (int(c2.x2), int(c2.y2)), (0, 255, 255), 3)

img = cv2.resize(img,(800,700))
cv2.imshow("with_line", img)
cv2.waitKey(0)
cv2.destroyWindow("with_line") #close the window

