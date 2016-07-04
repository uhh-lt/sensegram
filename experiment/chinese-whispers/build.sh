#!/bin/bash
mvn -DskipTests compile package dependency:build-classpath -Dmdep.outputFile=".dependency-jars"
