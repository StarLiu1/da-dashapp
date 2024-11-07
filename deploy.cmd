#!/bin/bash

# ----------------------
# KUDU Deployment Script
# Version: 1.0.17
# ----------------------

# Exit on any error
set -e

# Prerequisites
# -------------
# Verify node.js installed
if ! command -v node &> /dev/null; then
  echo "Missing node.js executable, please install node.js, or ensure it's in the current environment path."
  exit 1
fi

# Setup
# -----

ARTIFACTS="$(dirname "$0")/../artifacts"

if [ -z "$DEPLOYMENT_SOURCE" ]; then
  DEPLOYMENT_SOURCE="$(dirname "$0")/."
fi

if [ -z "$DEPLOYMENT_TARGET" ]; then
  DEPLOYMENT_TARGET="$ARTIFACTS/wwwroot"
fi

if [ -z "$NEXT_MANIFEST_PATH" ]; then
  NEXT_MANIFEST_PATH="$ARTIFACTS/manifest"
  if [ -z "$PREVIOUS_MANIFEST_PATH" ]; then
    PREVIOUS_MANIFEST_PATH="$ARTIFACTS/manifest"
  fi
fi

if [ -z "$KUDU_SYNC_CMD" ]; then
  # Install kudu sync
  echo "Installing Kudu Sync"
  npm install kudusync -g --silent
  if [ $? -ne 0 ]; then exit 1; fi

  # Set Kudu Sync command
  KUDU_SYNC_CMD="$HOME/.npm-global/bin/kuduSync"
fi

# Utility Functions
# -----------------

select_python_version() {
  if [ -n "$KUDU_SELECT_PYTHON_VERSION_CMD" ]; then
    $KUDU_SELECT_PYTHON_VERSION_CMD "$DEPLOYMENT_SOURCE" "$DEPLOYMENT_TARGET" "$DEPLOYMENT_TEMP"
    PYTHON_RUNTIME=$(<"$DEPLOYMENT_TEMP/__PYTHON_RUNTIME.tmp")
    PYTHON_VER=$(<"$DEPLOYMENT_TEMP/__PYTHON_VER.tmp")
    PYTHON_EXE=$(<"$DEPLOYMENT_TEMP/__PYTHON_EXE.tmp")
    PYTHON_ENV_MODULE=$(<"$DEPLOYMENT_TEMP/__PYTHON_ENV_MODULE.tmp")
  else
    PYTHON_RUNTIME="python-2.7"
    PYTHON_VER="2.7"
    PYTHON_EXE="/usr/bin/python2.7"
    PYTHON_ENV_MODULE="virtualenv"
  fi
}

# Deployment
# ----------

echo "Handling python deployment."

# 1. KuduSync
if [ "$IN_PLACE_DEPLOYMENT" != "1" ]; then
  $KUDU_SYNC_CMD -v 50 -f "$DEPLOYMENT_SOURCE" -t "$DEPLOYMENT_TARGET" -n "$NEXT_MANIFEST_PATH" -p "$PREVIOUS_MANIFEST_PATH" -i ".git;.hg;.deployment;deploy.sh"
fi

if [ ! -f "$DEPLOYMENT_TARGET/requirements.txt" ]; then
  exit 0
fi
if [ -f "$DEPLOYMENT_TARGET/.skipPythonDeployment" ]; then
  exit 0
fi

echo "Detected requirements.txt. You can skip Python-specific steps with a .skipPythonDeployment file."

# 2. Select Python version
select_python_version

cd "$DEPLOYMENT_TARGET"

# 3. Create virtual environment
if [ ! -f "env/azure.env.$PYTHON_RUNTIME.txt" ]; then
  if [ -d "env" ]; then
    echo "Deleting incompatible virtual environment."
    rm -rf "env"
  fi

  echo "Creating $PYTHON_RUNTIME virtual environment."
  $PYTHON_EXE -m $PYTHON_ENV_MODULE env
fi

# 4. Install packages
echo "Pip install requirements."
env/bin/pip install -r requirements.txt

# 5. Copy web.config
if [ -f "$DEPLOYMENT_SOURCE/web.$PYTHON_VER.config" ]; then
  echo "Overwriting web.config with web.$PYTHON_VER.config"
  cp -f "$DEPLOYMENT_SOURCE/web.$PYTHON_VER.config" "$DEPLOYMENT_TARGET/web.config"
fi

# 6. Django collectstatic
if [ -f "$DEPLOYMENT_TARGET/manage.py" ] && [ -d "$DEPLOYMENT_TARGET/env/lib/python$PYTHON_VER/site-packages/django" ]; then
  if [ ! -f "$DEPLOYMENT_TARGET/.skipDjango" ]; then
    echo "Collecting Django static files. You can skip Django-specific steps with a .skipDjango file."
    mkdir -p "$DEPLOYMENT_TARGET/static"
    env/bin/python manage.py collectstatic --noinput --clear
  fi
fi

echo "Finished successfully."
