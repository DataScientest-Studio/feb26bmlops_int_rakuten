all: dagshub setenv

dagshub: 
	echo "DAGSHUB_TOKEN=$(dagshub_token)" > .env.local
	echo "DAGSHUB_USER_TOKEN=$(dagshub_token)" >> .env.local
	echo "DAGSHUB_USER=$(dagshub_user)" >> .env.local

setenv:
	echo "UID=$$(id -u)" >> .env
	echo "GID=$$(id -g)" >> .env