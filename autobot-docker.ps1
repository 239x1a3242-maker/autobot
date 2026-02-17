# AutoBot Docker Management Script (PowerShell)
# Run with: .\autobot-docker.ps1

param(
    [Parameter(Mandatory=$false)]
    [string]$Command = "help"
)

# Configuration
$ComposeFile = "docker/docker-compose.yml"
$OverrideFile = "docker/docker-compose.override.yml"
$ProjectName = "autobot"

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Cyan"
$NC = "White"

function Write-ColorOutput {
    param([string]$Color, [string]$Message)
    Write-Host $Message -ForegroundColor $Color
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput $Blue "[INFO] $Message"
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput $Green "[SUCCESS] $Message"
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput $Yellow "[WARNING] $Message"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput $Red "[ERROR] $Message"
}

function Test-Docker {
    try {
        $null = Get-Command docker -ErrorAction Stop
        $null = Get-Command docker-compose -ErrorAction Stop
        return $true
    } catch {
        Write-Error "Docker or Docker Compose is not installed. Please install Docker Desktop first."
        return $false
    }
}

function Test-Nvidia {
    try {
        $null = Get-Command nvidia-smi -ErrorAction Stop
        Write-Info "NVIDIA GPU detected"
        return $true
    } catch {
        Write-Warning "NVIDIA GPU not detected - running in CPU-only mode"
        return $false
    }
}

function Invoke-Build {
    Write-Info "Building AutoBot Docker image..."
    try {
        docker-compose -f $ComposeFile build --no-cache
        Write-Success "Build completed"
        return $true
    } catch {
        Write-Error "Build failed: $($_.Exception.Message)"
        return $false
    }
}

function Invoke-Start {
    Write-Info "Starting AutoBot..."
    try {
        docker-compose -f $ComposeFile up -d
        Write-Success "AutoBot started successfully"
        Write-Info "Use 'docker-compose -f $ComposeFile logs -f autobot' to view logs"
        return $true
    } catch {
        Write-Error "Start failed: $($_.Exception.Message)"
        return $false
    }
}

function Invoke-Stop {
    Write-Info "Stopping AutoBot..."
    try {
        docker-compose -f $ComposeFile down
        Write-Success "AutoBot stopped"
        return $true
    } catch {
        Write-Error "Stop failed: $($_.Exception.Message)"
        return $false
    }
}

function Invoke-Restart {
    Write-Info "Restarting AutoBot..."
    try {
        docker-compose -f $ComposeFile restart
        Write-Success "AutoBot restarted"
        return $true
    } catch {
        Write-Error "Restart failed: $($_.Exception.Message)"
        return $false
    }
}

function Invoke-Logs {
    Write-Info "Showing AutoBot logs..."
    try {
        docker-compose -f $ComposeFile logs -f autobot
    } catch {
        Write-Error "Logs failed: $($_.Exception.Message)"
    }
}

function Invoke-Shell {
    Write-Info "Opening AutoBot shell..."
    try {
        docker-compose -f $ComposeFile exec autobot bash
    } catch {
        Write-Error "Shell access failed: $($_.Exception.Message)"
    }
}

function Get-Status {
    Write-Info "AutoBot status:"
    try {
        docker-compose -f $ComposeFile ps
    } catch {
        Write-Error "Status check failed: $($_.Exception.Message)"
    }
}

function Invoke-Clean {
    Write-Warning "This will remove all AutoBot containers, images, and volumes"
    $confirmation = Read-Host "Are you sure? (y/N)"
    if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
        Write-Info "Cleaning up AutoBot..."
        try {
            docker-compose -f $ComposeFile down -v --rmi all
            Write-Success "Cleanup completed"
            return $true
        } catch {
            Write-Error "Cleanup failed: $($_.Exception.Message)"
            return $false
        }
    }
    return $false
}

function Invoke-Test {
    Write-Info "Running AutoBot tests..."
    try {
        docker-compose -f $ComposeFile exec autobot python3 test_real_world_automation.py
    } catch {
        Write-Error "Test execution failed: $($_.Exception.Message)"
    }
}

function Invoke-Dev {
    Write-Info "Starting AutoBot in development mode..."
    try {
        docker-compose -f $ComposeFile -f $OverrideFile up -d
        Write-Success "AutoBot development mode started"
        return $true
    } catch {
        Write-Error "Development mode start failed: $($_.Exception.Message)"
        return $false
    }
}

function Show-Usage {
    Write-Host "AutoBot Docker Management Script (PowerShell)"
    Write-Host ""
    Write-Host "Usage: .\autobot-docker.ps1 [-Command] <command>"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build     Build AutoBot Docker image"
    Write-Host "  start     Start AutoBot"
    Write-Host "  stop      Stop AutoBot"
    Write-Host "  restart   Restart AutoBot"
    Write-Host "  logs      Show AutoBot logs"
    Write-Host "  shell     Open AutoBot shell"
    Write-Host "  status    Show AutoBot status"
    Write-Host "  test      Run AutoBot tests"
    Write-Host "  dev       Start in development mode"
    Write-Host "  clean     Remove all AutoBot containers and images"
    Write-Host "  help      Show this help"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\autobot-docker.ps1 -Command build"
    Write-Host "  .\autobot-docker.ps1 -Command start"
    Write-Host "  .\autobot-docker.ps1 -Command logs"
    Write-Host "  .\autobot-docker.ps1 -Command shell"
    Write-Host "  .\autobot-docker.ps1 -Command dev"
}

# Main script logic
switch ($Command) {
    "build" {
        if (Test-Docker) {
            Test-Nvidia | Out-Null
            Invoke-Build
        }
    }
    "start" {
        if (Test-Docker) {
            Invoke-Start
        }
    }
    "stop" {
        if (Test-Docker) {
            Invoke-Stop
        }
    }
    "restart" {
        if (Test-Docker) {
            Invoke-Restart
        }
    }
    "logs" {
        if (Test-Docker) {
            Invoke-Logs
        }
    }
    "shell" {
        if (Test-Docker) {
            Invoke-Shell
        }
    }
    "status" {
        if (Test-Docker) {
            Get-Status
        }
    }
    "test" {
        if (Test-Docker) {
            Invoke-Test
        }
    }
    "dev" {
        if (Test-Docker) {
            Invoke-Dev
        }
    }
    "clean" {
        if (Test-Docker) {
            Invoke-Clean
        }
    }
    "help" {
        Show-Usage
    }
    default {
        Write-Error "Unknown command: $Command"
        Write-Host ""
        Show-Usage
        exit 1
    }
}