package tools

const (
	ResultPath = "/BIGDATA1/acict_zguan_1/zpltys/graphs/generate/result/"

	//PatternPath = "../test_data/pattern.txt"
	PatternPath              = "pattern.txt"
	GraphSimulationTypeModel = 100

	RPCSendSize = 100000

	ConnPoolSize       = 16
	MasterConnPoolSize = 2048

	UseCuda = true
)

var dataPath string
var partitionStrategy string

func SetDataPath(path string) {
	if path[len(path)-1] != '/' {
		dataPath = path + "/"
	} else {
		dataPath = path
	}
}

func GetConfigPath(partitionNum int) string {
	return "config.txt"
	//return "config2.txt"
}

func GetDataPath() string {
	return dataPath
}
