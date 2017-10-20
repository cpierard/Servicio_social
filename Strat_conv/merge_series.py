import pathlib
from dedalus.tools import post
set_paths = list(pathlib.Path("ugm_28_1hr/").glob("ugm_28_1hr_s*.h5"))
post.merge_sets("ugm_28_1hr/ugm_28_1hr.h5", set_paths, cleanup=True)
