#!/usr/bin/env groovy
// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS
@Library('rocJenkins@noDocker') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 1 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])


////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;

rcclTestsCI:
{
    def rcclTests = new rocProject('rcclTests')
    // customize for project
    rcclTests.paths.build_command = './install.sh'

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['RCCL'], rcclTests)

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        def command = """#!/usr/bin/env bash
                  set -x
                  rm -rf rccl
                  git clone https://github.com/ROCmSoftwarePlatform/rccl
                  cd rccl
                  export RCCL_PATH=${WORKSPACE}/rccl/rccl-install
                  ./install.sh -i --prefix=\$RCCL_PATH
                  cd ..
                  cd ${project.paths.project_build_prefix}
                  ${project.paths.build_command} --rccl_home=\$RCCL_PATH
                """
	  sh command
    }
    def testCommand =
    {
        platform, project->

        def command = """#!/usr/bin/env bash
                set -x
                LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${WORKSPACE}/rccl/rccl-install/lib/ python3 -m pytest -k "not MPI and not AllGather and not Reduce and not ReduceScatter" --junitxml=./testreport.xml
            """

        sh command
        junit "${project.paths.project_build_prefix}/*.xml"
    }

    def packageCommand =
    {
        platform, project->

        def command = """
                      """
    }

    buildProjectNoDocker(rcclTests, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}
