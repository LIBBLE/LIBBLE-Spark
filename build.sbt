
name := "LIBBLE-Spark"

version := "0.0.1"

scalaVersion := "2.10.5"

artifactName := { (sv: ScalaVersion, module: ModuleID, artifact: Artifact) =>
  artifact.name + "-" + module.revision + "." + artifact.extension
}

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.5.1"

resolvers ++= Seq(
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)