name := "libble-spark"

version := "1.0.1"

organization := "libble"

scalaVersion := "2.11.7"


artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) =>
  artifact.name + "-" + module.revision + "." + artifact.extension
}



libraryDependencies += "org.apache.spark"%%"spark-core"%"2.0.1"

libraryDependencies += "org.scalatest"%%"scalatest"%"3.0.0"

