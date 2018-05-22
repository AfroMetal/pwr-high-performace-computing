# Pohilg-Hellman algorithm for DLP

>High Performance Computing task5

## BUILD

1. Navigate to `cmake-build-debug/`.
2. Run `$ make`.
3. `hpc_pohlig` bin file should be created.

## RUN

### To run with data generation

```bash
$ ./hpc_pohlig pi-bits amount-of-pi
``` 

### To run with complete input

```bash
$ ./hpc_pohlig P alpha beta Q (pi ei)...
``` 

### Running on arguments from file

```bash
$ ./hpc_pohlig $(cat /path/to/file.txt)
``` 

### Example input

`file.txt`:

```
251

71

210

2

5
3
```

or as commandline arguments directly:

```bash
$ ./hpc_pohlig 251 71 210 2 5 3
```
