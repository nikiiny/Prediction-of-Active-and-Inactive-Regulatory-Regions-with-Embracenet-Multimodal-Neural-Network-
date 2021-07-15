from .dataload import Load_Create_Task
from .dataprepare import Build_DataLoader_Pipeline

TASKS = ['active_E_vs_inactive_E', 'active_P_vs_inactive_P', 
                        'active_E_vs_active_P', 'inactive_E_vs_inactive_P',
                        'active_EP_vs_inactive_rest']

CELL_LINES = ['A549','GM12878', 'H1', 'HEK293', 'HEPG2', 'K562', 'MCF7']

__all__ = ['Load_Create_Task', 'Build_DataLoader_Pipeline']
