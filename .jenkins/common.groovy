// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    String hipclangArgs = jobName.contains('hipclang') ? '--hip-clang' : ''
    def getRCCL = auxiliary.getLibrary('rccl',platform.jenkinsLabel,'develop')

    def command = """#!/usr/bin/env bash
                set -x
                ${getRCCL}
                ${auxiliary.exitIfNotSuccess()}
                cd ${project.paths.project_build_prefix}
                cmake \
                    -DCMAKE_CXX_COMPILER=/opt/rocm/hip/bin/hipcc \
                    -S . -B build
                make -C build -j\$(nproc)
                ${auxiliary.exitIfNotSuccess()}
            """

    platform.runCommand(this,command)
}

def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
		python3 -m pip install --upgrade pytest
		python3 -m pytest --version
		python3 -m pytest -k "not MPI and not host and not fine" --verbose --junitxml=./testreport.xml
            """

   platform.runCommand(this, command)
   junit "${project.paths.project_build_prefix}/*.xml"
}

return this
