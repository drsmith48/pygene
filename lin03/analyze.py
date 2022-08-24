from pathlib import Path
import dataclasses
import re
import struct
import os
from typing import Union, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclasses.dataclass
class LinearScan(object):
    directory: Union[Path,str] = Path()
    scan_index: int = 0  # 0 (default) gives `scanfiles0000/`

    def __post_init__(self):
        self.directory = Path(self.directory)
        assert self.directory.exists(), f"{self.directory} does not exist"
        self.scandir = self.directory / f'scanfiles{self.scan_index:04d}'
        assert self.scandir.exists(), f"{self.scandir} does not exist"
        self.scanlog_file = self.scandir / 'scan.log'
        assert self.scanlog_file.exists(), f"{self.scanlog_file} does not exist"
        self.short_path = self.directory.relative_to(Path.home()/'gene/tigress/lin03')
        self.scanlog_df = None

    def _read_scanlog(self, verbose=True):
        if verbose: print(f"Reading scan.log: {self.scanlog_file}")
        with self.scanlog_file.open('r') as scanlog_file:
            re_header = re.compile('\| \w+')
            for i_line, line in enumerate(scanlog_file):
                if i_line==0:
                    self.scan_vars = [s.removeprefix('| ') for s in re_header.findall(line)]
                    self.n_vars = len(self.scan_vars)
                    assert self.n_vars >= 1
                    re_run = '([0-9]+)\s+\|'
                    re_run += '\s+([0-9.e+-]+)\s+\|' * self.n_vars
                    re_run += '\s+([0-9.Nae+-]+)\s+([0-9.Nae+-]+)'
                    re_run = re.compile(re_run)
                    scanlog_dict = {}
                    for var_name in self.scan_vars:
                        scanlog_dict[var_name] = np.empty(0)
                    scanlog_dict['gamma'] = np.empty(0)
                    scanlog_dict['omega'] = np.empty(0)
                    n_runs = 0
                else:
                    match = re_run.match(line)
                    assert match, f"Match failed: {line}"
                    groups = match.groups()
                    assert len(groups) == self.n_vars + 3
                    for i_var, var in enumerate(groups[1:-2]):
                        scanlog_dict[self.scan_vars[i_var]] = np.append(
                            scanlog_dict[self.scan_vars[i_var]],
                            float(var),
                        )
                    scanlog_dict['gamma'] = np.append(
                        scanlog_dict['gamma'],
                        float(groups[-2]),
                    )
                    scanlog_dict['omega'] = np.append(
                        scanlog_dict['omega'],
                        float(groups[-1]),
                    )
                    n_runs += 1
        scanlog_df = pd.DataFrame(scanlog_dict)
        scanlog_df.index += 1
        for var in self.scan_vars:
            column = scanlog_df[var].to_numpy()
            if np.all([item.is_integer() for item in column]):
                scanlog_df = scanlog_df.astype({var:int})
        cols = scanlog_df.columns.tolist()
        for key_vars in ['nz0', 'hyp_z', 'nky0']:
            if key_vars in cols:
                cols.insert(0, cols.pop(cols.index(key_vars)))
        scanlog_df = scanlog_df[cols]
        self.scanlog_df = scanlog_df
        self.scan_vars = self.scanlog_df.columns.tolist()[:-2]
        if verbose:
            print(self.scanlog_df)
            print(self.scan_vars)

    def plot_omega(self, logx: bool = False, verbose = True):
        if self.scanlog_df is None:
            self._read_scanlog(verbose=verbose)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,2))
        for i_plot, quantity in enumerate(['gamma', 'omega']):
            plt.sca(axes.flat[i_plot])
            plt.xlabel(self.scan_vars[0])
            plt.ylabel(quantity)
            plt.title(self.short_path)
            if self.n_vars == 3:
                v2_values = pd.unique(self.scanlog_df[self.scan_vars[1]])
                v3_values = pd.unique(self.scanlog_df[self.scan_vars[2]])
                for v2 in v2_values:
                    for v3 in v3_values:
                        sub_df = self.scanlog_df[
                            (self.scanlog_df[self.scan_vars[1]]==v2) & (self.scanlog_df[self.scan_vars[2]]==v3)
                        ]
                        plt.plot(sub_df[self.scan_vars[0]], sub_df[quantity], 'x-',
                                 label=f"{self.scan_vars[1]}={v2}, {self.scan_vars[2]}={v3}")
            elif self.n_vars == 2:
                v2_values = pd.unique(self.scanlog_df[self.scan_vars[1]])
                for v2 in v2_values:
                    sub_df = self.scanlog_df.loc[
                        self.scanlog_df[self.scan_vars[1]] == v2,
                    ]
                    plt.plot(sub_df[self.scan_vars[0]], sub_df[quantity], 'x-',
                             label=f"{self.scan_vars[1]}={v2}")
            else:
                pass
            if logx:
                plt.xscale('log')
            ylim = np.array(plt.gca().get_ylim())
            ylim[0] = np.min([0, ylim[0]])
            ylim[1] = np.max([0, ylim[1]])
            plt.ylim(ylim)
            plt.legend(fontsize='small')
        plt.tight_layout()

    def _get_mode(self, index: int = None) -> dict:
        assert index > 0
        parameters_file = self.scandir / f"parameters_{index:04d}"
        assert parameters_file.exists(), f"{parameters_file} does not exist"
        field_file = self.scandir / f"field_{index:04d}"
        assert field_file.exists(), f"{field_file} does not exist"
        print(f"Parameters file: {parameters_file}")
        print(f"Field file: {field_file}")
        parameters = {
            'nx0': None,
            'nz0': None,
            'PRECISION': None,
            'ENDIANNESS': None,
        }
        with parameters_file.open('r') as pfile:
            for line in pfile:
                for key in parameters:
                    if key in line:
                        match = re.match('\w+\s*=\s*(\w+)', line)
                        parameters[key] = match.group(1)
                        if key in ['nx0','nz0']:
                            parameters[key] = int(parameters[key])
        for value in parameters.values():
            assert value is not None
        nx0 = parameters['nx0']
        nz0 = parameters['nz0']
        # ballooning grid
        n_ballooning_grid = (nx0-1) * nz0
        ballooning_grid = np.empty(n_ballooning_grid)
        delz = 2 * np.pi / nz0
        zgrid = np.linspace(-np.pi, np.pi-delz, nz0)
        for i in range(nx0-1):
            ballooning_grid[i*nz0:(i+1)*nz0] = (
                2 * (i-(nx0-1)//2) + zgrid/np.pi
            )
        intsize = 4
        realsize = 8 if parameters['PRECISION']=='DOUBLE' else 4
        complexsize = 2*realsize
        n_data_points = nx0 * nz0
        entrysize = n_data_points * complexsize
        if parameters['ENDIANNESS']=='BIG':
            nprt=(np.dtype(np.float64)).newbyteorder()
            npct=(np.dtype(np.complex128)).newbyteorder()
            fmt = '>idi'
        else:
            nprt=np.dtype(np.float64)
            npct=np.dtype(np.complex128)
            fmt = '=idi'
        te = struct.Struct(fmt)
        tesize = te.size
        leapfld = 2 * (entrysize+2*intsize)
        filesize = os.path.getsize(field_file.as_posix())
        n_time = filesize // (leapfld+tesize)
        fields = {
            'phi' : np.empty(n_data_points, dtype=npct),
            'apar' : np.empty(n_data_points, dtype=npct),
        }
        for i_field, field_name in enumerate(fields):
            offset = (n_time-1) * (tesize + leapfld) + \
                     i_field * (entrysize + 2 * intsize) + \
                     intsize + tesize
            with field_file.open('rb') as f:
                f.seek(offset)
                data = np.fromfile(f,  # shape nz*nx
                                   count=n_data_points,
                                   dtype=npct)
            assert data.size == nx0 * nz0
            data = data.reshape((nz0, nx0)).transpose()  # reshape to nx, nz
            data = np.roll(
                data,
                nx0//2-1,
                axis=0,
            )  # roll x dimension
            data = np.reshape(
                data[0:-1,:],
                (nx0-1)*nz0,
                order='C',
            )  # reshape to ballooning grid
            assert data.size == ballooning_grid.size
            fields[field_name] = data
        max_amplitude = np.max(
            [np.max(np.abs(fields[field_name])) for field_name in fields]
        )
        for field_name in fields:
            fields[field_name] /= max_amplitude
        return fields, ballooning_grid

    def plot_mode(self, index: int|Iterable = None, fields: str|Iterable = None, **kwargs):
        if self.scanlog_df is None:
            self._read_scanlog()
        bool_arr = None
        if index is None:
            assert kwargs
            for i_var, var in enumerate(self.scan_vars):
                if var not in kwargs: continue
                if bool_arr is None:
                    bool_arr = self.scanlog_df[var] == kwargs[var]
                else:
                    bool_arr = bool_arr & (self.scanlog_df[var] == kwargs[var])
            sub_df = self.scanlog_df[bool_arr]
        else:
            if isinstance(index, int):
                index = [index]
            sub_df = self.scanlog_df.index[index]
        for index in sub_df.index:
            row_text = ', '.join(
                [f"{var_name}={self.scanlog_df.loc[index, var_name]}" for var_name in self.scan_vars]
            )
            fields_dict, ballooning_grid = self._get_mode(index=index)
            if fields is None:
                fields = list(fields_dict.keys())
            elif isinstance(fields, str):
                fields = [fields]
            fig, axes = plt.subplots(
                nrows=len(fields),
                ncols=2,
                figsize=(10,2*len(fields)),
                sharex='col',
            )
            for i_field, field_name in enumerate(fields):
                for i_col in [0,1]:
                    plt.sca(axes.flat[2*i_field + i_col])
                    plt.title(f"{self.short_path} | Run index {index:04d}")
                    plt.plot(
                        ballooning_grid,
                        np.real(fields_dict[field_name]),
                        label=f"Re({field_name})",
                    )
                    plt.plot(
                        ballooning_grid,
                        np.abs(fields_dict[field_name]),
                        label=f"Abs({field_name})",
                    )
                    plt.legend(fontsize='small')
                    if i_field == len(fields_dict)-1:
                        plt.xlabel('Ballooning angle (rad/pi)')
                    if i_col == 1:
                        plt.xlim(-2,2)
                    plt.annotate(
                        row_text,
                        (0.01, 0.03),
                        xycoords='axes fraction',
                        fontsize='small',
                    )
            plt.tight_layout()


if __name__=='__main__':
    dir = Path.home()/'gene/tigress/lin03/eq21/pn60/nzhypz'
    scan = LinearScan(
        directory=dir,
    )
    scan.plot_omega(logx=True)
    scan.plot_mode(hyp_z=2)
    plt.show()