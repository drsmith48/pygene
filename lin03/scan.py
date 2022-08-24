from pathlib import Path
import shutil
import subprocess
import fileinput

from setup import EQUILIBRIA, PSINORM_VALUES, TIGRESS_DIR, DEFAULTS_DIR, VERSION

class _Scan(object):

    def __init__(
            self,
            equilibrium: str = None,
            psinorm: str = None,
            scan_name: str = None,
            overwrite: bool = False,
    ):
        # ensure equilibrium and psinorm are valid
        assert equilibrium in EQUILIBRIA, f"Invalid equilibrium: {equilibrium}"
        assert psinorm in PSINORM_VALUES, f"Invalid psinorm: {psinorm}"

        self.equilibrium = equilibrium
        self.psinorm = psinorm
        self.scan_name = scan_name
        self.overwrite = overwrite

        assert self.scan_name in self.__class__.__name__

        # create equilibrium, psinorm, and scan directories
        self.scan_dir = TIGRESS_DIR / self.equilibrium / self.psinorm / self.scan_name
        self.scan_dir.mkdir(exist_ok=True, parents=True)

        # ensure `parameters-default` exists
        self.parameters_default_file = DEFAULTS_DIR / self.equilibrium / self.psinorm / 'parameters-default'
        assert self.parameters_default_file.exists()

        # ensure launcher template exists
        self.launcher_template = Path('./launcher.template.sh').resolve()
        assert self.launcher_template.exists()

        # copy parameters and launcher to scan_dir
        self.parameters_file = self.scan_dir / 'parameters'
        self.launcher_file = self.scan_dir / 'launcher.sh'
        if not self.overwrite and (self.parameters_file.exists() or self.launcher_file.exists()):
            raise FileExistsError
        shutil.copy(self.parameters_default_file, self.parameters_file)
        shutil.copy(self.launcher_template, self.launcher_file)

        self.job_name = '-'.join(
            [VERSION, self.equilibrium, self.psinorm, self.scan_name]
        )
        print(self.job_name)

        for line in fileinput.input(self.launcher_file, inplace=True):
            if '--job-name' in line:
                line = f'#SBATCH --job-name={self.job_name}\n'
            print(line, end='')

        # comment out `curv_factor` in parameters
        # for line in fileinput.input(self.parameters_file, inplace=True):
        #     if line.startswith('curv_factor'):
        #         line = line.replace('curv_factor', '!curv_factor')
        #     print(line, end='')

    def submit(self):
        ret = subprocess.run(
            f"cd {self.scan_dir.as_posix()} && echo PWD: $PWD && sbatch launcher.sh",
            timeout=10,
            capture_output=True,
            text=True,
            shell=True,
        )
        ret.check_returncode()
        print(ret.stdout)


class Scan_nxnzhypz(_Scan):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        self.scan_name = 'nxnzhypz'
        super().__init__(scan_name=self.scan_name, *args, **kwargs)
        for line in fileinput.input(self.parameters_file, inplace=True):
            if line.startswith('nx0'):
                line = 'nx0 = 64  !scanlist: 32, 48, 64\n'
            elif line.startswith('nz0'):
                line = 'nz0 = 128  !scanlist: 64, 96, 128\n'
            elif line.startswith('hyp_z'):
                line = 'hyp_z = 20  !scanlist: 5, 20\n'
            elif line.startswith('kymin'):
                line = 'kymin = 0.3\n'
            elif line.startswith('timelim'):
                line = 'timelim = 85000\n'
            print(line, end='')


class Scan_nxnzhypz2(_Scan):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        self.scan_name = 'nxnzhypz2'
        super().__init__(scan_name=self.scan_name, *args, **kwargs)
        for line in fileinput.input(self.parameters_file, inplace=True):
            if line.startswith('nx0'):
                line = 'nx0 = 64  !scanlist: 64, 32\n'
            elif line.startswith('nz0'):
                line = 'nz0 = 128  !scanlist: 128, 64\n'
            elif line.startswith('hyp_z'):
                line = 'hyp_z = 20  !scanlist: 0.5, 2, 8, 32\n'
            elif line.startswith('kymin'):
                line = 'kymin = 0.3\n'
            elif line.startswith('timelim'):
                line = 'timelim = 85000\n'
            print(line, end='')


class Scan_nzhypz(_Scan):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        self.scan_name = 'nzhypz'
        super().__init__(scan_name=self.scan_name, *args, **kwargs)
        for line in fileinput.input(self.parameters_file, inplace=True):
            if line.startswith('nx0'):
                line = 'nx0 = 64\n'
            elif line.startswith('nz0'):
                line = 'nz0 = 128  !scanlist: 128, 64, 32, 16\n'
            elif line.startswith('hyp_z'):
                line = 'hyp_z = 20  !scanlist: 0.1, 0.5, 2, 8, 32\n'
            elif line.startswith('kymin'):
                line = 'kymin = 0.3\n'
            elif line.startswith('timelim'):
                line = 'timelim = 85000\n'
            print(line, end='')


class Scan_nzhypz2(_Scan):

    def __init__(self, *args, **kwargs):
        self.scan_name = 'nzhypz2'
        super().__init__(scan_name=self.scan_name, *args, **kwargs)
        for line in fileinput.input(self.parameters_file, inplace=True):
            if line.startswith('nx0'):
                line = 'nx0 = 64\n'
            elif line.startswith('nz0'):
                line = 'nz0 = 128  !scanlist: 128, 64, 32\n'
            elif line.startswith('hyp_z'):
                line = 'hyp_z = 20  !scanlist: 10, 20, 30, 40, 60\n'
            elif line.startswith('kymin'):
                line = 'kymin = 0.3\n'
            elif line.startswith('timelim'):
                line = 'timelim = 85000\n'
            print(line, end='')


class Scan_kynzhypz(_Scan):

    def __init__(self, *args, **kwargs):
        self.scan_name = 'kynzhypz'
        super().__init__(scan_name=self.scan_name, *args, **kwargs)
        for line in fileinput.input(self.parameters_file, inplace=True):
            if line.startswith('nx0'):
                line = 'nx0 = 64\n'
            elif line.startswith('nz0'):
                line = 'nz0 = 128 !scanlist: 128, 64\n'
            elif line.startswith('hyp_z'):
                line = 'hyp_z = 60 !scanlist: 60, 90\n'
            elif line.startswith('kymin'):
                line = 'kymin = 0.3 !scanlist: 0.15, 0.3, 0.6\n'
            elif line.startswith('timelim'):
                line = 'timelim = 85000\n'
            print(line, end='')


if __name__ == '__main__':
    for eq in EQUILIBRIA:
        for pn in PSINORM_VALUES:
            scan = Scan_kynzhypz(
                equilibrium=eq,
                psinorm=pn,
            )
            scan.submit()