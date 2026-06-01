import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Alert,
  AppBar,
  Box,
  Button,
  Card,
  CardContent,
  CardMedia,
  Chip,
  Container,
  CssBaseline,
  Divider,
  Link,
  Paper,
  Stack,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  ThemeProvider,
  ToggleButton,
  ToggleButtonGroup,
  Toolbar,
  Typography,
  createTheme,
} from "@mui/material";
import ArticleOutlinedIcon from "@mui/icons-material/ArticleOutlined";
import BiotechOutlinedIcon from "@mui/icons-material/BiotechOutlined";
import DataObjectOutlinedIcon from "@mui/icons-material/DataObjectOutlined";
import FactCheckOutlinedIcon from "@mui/icons-material/FactCheckOutlined";
import ImageSearchOutlinedIcon from "@mui/icons-material/ImageSearchOutlined";
import InsightsOutlinedIcon from "@mui/icons-material/InsightsOutlined";
import QueryStatsOutlinedIcon from "@mui/icons-material/QueryStatsOutlined";
import UploadFileOutlinedIcon from "@mui/icons-material/UploadFileOutlined";
import "./styles.css";

const apiUrl = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#155e63",
      dark: "#0d3f43",
    },
    secondary: {
      main: "#9a513f",
    },
    background: {
      default: "#eef2ef",
      paper: "#ffffff",
    },
    text: {
      primary: "#16211f",
      secondary: "#5c6763",
    },
  },
  shape: {
    borderRadius: 14,
  },
  typography: {
    fontFamily: '"Aptos", "Segoe UI", sans-serif',
    h1: {
      fontSize: "2rem",
      fontWeight: 800,
      letterSpacing: "-0.03em",
    },
    h2: {
      fontSize: "1.15rem",
      fontWeight: 800,
    },
    button: {
      fontWeight: 800,
      textTransform: "none",
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          border: "1px solid rgba(22, 33, 31, 0.08)",
          boxShadow: "0 18px 50px rgba(22, 33, 31, 0.08)",
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        head: {
          color: "#5c6763",
          fontSize: 12,
          fontWeight: 800,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
        },
      },
    },
  },
});

const artifactLabels = {
  original_image: "Исходное изображение",
  mask_image: "Маска ядра",
  overlay_image: "Наложение маски",
  lobe_foreground_image: "Foreground сегментов",
  lobe_boundary_image: "Границы сегментов",
  lobe_components_image: "Компоненты сегментов",
  lobe_overlay_image: "Наложение сегментов",
  report_json: "JSON-отчет",
  report_markdown: "Markdown-отчет",
  log_file: "Лог анализа",
};

const featureLabels = {
  nucleus_area_px: "Площадь ядра, px",
  nucleus_perimeter_px: "Периметр ядра, px",
  nucleus_circularity: "Округлость",
  nucleus_solidity: "Плотность / solidity",
  nucleus_eccentricity: "Эксцентриситет",
  nucleus_extent: "Заполнение bbox",
  nucleus_orientation_degrees: "Ориентация, градусы",
  nucleus_major_axis_length_px: "Большая ось, px",
  nucleus_minor_axis_length_px: "Малая ось, px",
  nucleus_aspect_ratio: "Соотношение сторон",
  nucleus_convex_area_px: "Выпуклая площадь, px",
  nucleus_filled_area_px: "Заполненная площадь, px",
  nucleus_bbox_width_px: "Ширина bbox, px",
  nucleus_bbox_height_px: "Высота bbox, px",
  nucleus_segments: "Сегменты ядра",
  segment_area_mean_px: "Средняя площадь сегмента, px",
  segment_area_min_px: "Минимальная площадь сегмента, px",
  segment_area_max_px: "Максимальная площадь сегмента, px",
  mask_foreground_fraction: "Доля foreground",
};

const metadataLabels = {
  pipeline_version: "Версия пайплайна",
  segmenter_name: "Сегментатор ядра",
  lobe_counter_name: "Счетчик сегментов",
  classifier_name: "Классификатор",
  postprocessing: "Постобработка",
};

const postprocessingLabels = {
  segment_min_area_px: "Мин. площадь сегмента, px",
  segment_min_peak_distance_px: "Мин. расстояние пиков, px",
  watershed_compactness: "Watershed compactness",
  lobe_foreground_threshold: "Порог foreground сегментов",
  lobe_boundary_threshold: "Порог границ сегментов",
  lobe_min_segment_area_px: "Мин. площадь доли, px",
  yolo_lobe_confidence: "YOLO confidence",
  yolo_lobe_iou: "YOLO IoU",
  yolo_lobe_image_size: "YOLO размер изображения",
  yolo_lobe_min_mask_area_px: "YOLO мин. площадь маски, px",
  border_margin_px: "Отступ от края, px",
  cluster_distance_fraction: "Дистанция кластеров",
  cluster_min_area_ratio: "Мин. площадь кластера",
};

function artifactUrl(path) {
  if (!path) {
    return "";
  }
  return `${apiUrl}/artifacts/${path}`;
}

function formatValue(value) {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toString() : value.toFixed(3);
  }
  if (typeof value === "boolean") {
    return value ? "Да" : "Нет";
  }
  return String(value);
}

function entriesToRows(source, labels = {}, prefix = "") {
  if (!source) {
    return [];
  }
  return Object.entries(source).flatMap(([key, value]) => {
    const label = labels[key] ?? key;
    const name = prefix ? `${prefix} · ${label}` : label;
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const nestedLabels = key === "postprocessing" ? postprocessingLabels : {};
      return entriesToRows(value, nestedLabels, name);
    }
    return [{ key: prefix ? `${prefix}.${key}` : key, name, value: formatValue(value) }];
  });
}

function DetailTable({ rows, emptyText = "Нет данных" }) {
  if (!rows.length) {
    return (
      <Typography color="text.secondary" variant="body2">
        {emptyText}
      </Typography>
    );
  }

  return (
    <TableContainer component={Paper} variant="outlined">
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Параметр</TableCell>
            <TableCell>Значение</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.key}>
              <TableCell sx={{ width: "42%", fontWeight: 700 }}>{row.name}</TableCell>
              <TableCell sx={{ overflowWrap: "anywhere" }}>{row.value}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [analysisMode, setAnalysisMode] = useState("pipeline");
  const [detailsTab, setDetailsTab] = useState("features");

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const reportUrl = result ? `${apiUrl}/analysis/${result.analysis_id}/report` : "";
  const logsUrl = result ? `${apiUrl}/analysis/${result.analysis_id}/logs` : "";
  const visualArtifacts = result
    ? Object.entries(result.artifacts)
        .filter(([key, path]) => path && key.endsWith("_image"))
        .map(([key, path]) => ({
          key,
          label: artifactLabels[key] ?? key,
          src: artifactUrl(path),
        }))
    : [];
  const artifactRows = result
    ? Object.entries(result.artifacts)
        .filter(([, path]) => path)
        .map(([key, path]) => ({
          key,
          name: artifactLabels[key] ?? key,
          value: path,
          href: artifactUrl(path),
        }))
    : [];

  async function runAnalysis(event) {
    event.preventDefault();
    if (!file) {
      setError("Выберите изображение для анализа.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const endpoints = {
        pipeline: "/analysis",
        yolo: "/analysis/yolo",
      };
      const endpoint = endpoints[analysisMode] ?? "/analysis";
      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.message ?? "Analysis failed.");
      }
      setResult(payload);
      setDetailsTab("features");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  const summaryCards = result
    ? [
        {
          label: "Классификация",
          value: result.classification.label,
          icon: <FactCheckOutlinedIcon color="primary" />,
        },
        {
          label: "Сегменты ядра",
          value: result.features.nucleus_segments,
          icon: <QueryStatsOutlinedIcon color="primary" />,
        },
        {
          label: "Площадь ядра",
          value: `${result.features.nucleus_area_px}px`,
          icon: <BiotechOutlinedIcon color="primary" />,
        },
        {
          label: "Счетчик сегментов",
          value: result.metadata.lobe_counter_name,
          icon: <InsightsOutlinedIcon color="primary" />,
        },
      ]
    : [];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box className="app-background">
        <AppBar color="inherit" elevation={0} position="sticky" sx={{ borderBottom: 1, borderColor: "divider" }}>
          <Toolbar sx={{ gap: 2 }}>
            <BiotechOutlinedIcon color="primary" />
            <Box sx={{ flex: 1 }}>
              <Typography fontWeight={900}>Neutrophil Analysis Console</Typography>
              <Typography color="text.secondary" variant="caption">
                U-Net nucleus segmentation · lobe counting · rule-based classification
              </Typography>
            </Box>
            <Chip
              color={result?.status === "completed" ? "success" : "default"}
              label={result?.status ?? "idle"}
              size="small"
              sx={{ fontWeight: 800, textTransform: "uppercase" }}
            />
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ py: 3 }}>
          <Box
            sx={{
              alignItems: "flex-start",
              display: "grid",
              gap: 3,
              gridTemplateColumns: { md: "minmax(320px, 0.36fr) minmax(0, 1fr)", xs: "1fr" },
            }}
          >
            <Box>
              <Card>
                <CardContent>
                  <Stack component="form" onSubmit={runAnalysis} spacing={2.5}>
                    <Box>
                      <Typography variant="h2">Новый анализ</Typography>
                      <Typography color="text.secondary" sx={{ mt: 0.5 }} variant="body2">
                        Загрузите изображение одного нейтрофила и выберите режим подсчета сегментов.
                      </Typography>
                    </Box>

                    <Button
                      component="label"
                      fullWidth
                      size="large"
                      startIcon={<UploadFileOutlinedIcon />}
                      sx={{ minHeight: 92, borderStyle: "dashed" }}
                      variant="outlined"
                    >
                      {file ? file.name : "Выбрать изображение"}
                      <input
                        accept="image/png,image/jpeg,image/tiff,image/bmp"
                        hidden
                        onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                        type="file"
                      />
                    </Button>

                    {previewUrl && (
                      <Card variant="outlined">
                        <CardMedia
                          alt="Предпросмотр выбранного изображения"
                          component="img"
                          image={previewUrl}
                          sx={{ bgcolor: "grey.100", height: 220, objectFit: "contain" }}
                        />
                      </Card>
                    )}

                    <Box>
                      <Typography fontWeight={800} gutterBottom variant="body2">
                        Режим анализа
                      </Typography>
                      <ToggleButtonGroup
                        color="primary"
                        exclusive
                        fullWidth
                        onChange={(_, value) => value && setAnalysisMode(value)}
                        size="small"
                        value={analysisMode}
                      >
                        <ToggleButton value="pipeline">U-Net pipeline</ToggleButton>
                        <ToggleButton value="yolo">YOLO lobes</ToggleButton>
                      </ToggleButtonGroup>
                    </Box>

                    <Button
                      disabled={isLoading}
                      fullWidth
                      size="large"
                      startIcon={<ImageSearchOutlinedIcon />}
                      type="submit"
                      variant="contained"
                    >
                      {isLoading ? "Анализ выполняется" : "Запустить анализ"}
                    </Button>

                    {error && <Alert severity="error">{error}</Alert>}
                  </Stack>
                </CardContent>
              </Card>
            </Box>

            <Box>
              <Stack spacing={3}>
                <Paper sx={{ p: 3 }} variant="outlined">
                  <Stack
                    direction={{ md: "row", xs: "column" }}
                    spacing={2}
                    sx={{ alignItems: { md: "center", xs: "flex-start" } }}
                  >
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="h1">Результат анализа ядра</Typography>
                    </Box>
                    {result && (
                      <Stack direction="row" spacing={1}>
                        <Button
                          component={Link}
                          href={reportUrl}
                          startIcon={<ArticleOutlinedIcon />}
                          target="_blank"
                          variant="outlined"
                        >
                          Отчет
                        </Button>
                        <Button component={Link} href={logsUrl} target="_blank" variant="text">
                          Логи
                        </Button>
                      </Stack>
                    )}
                  </Stack>
                </Paper>

                {result ? (
                  <>
                    <Box
                      sx={{
                        display: "grid",
                        gap: 2,
                        gridTemplateColumns: {
                          lg: "repeat(4, minmax(0, 1fr))",
                          sm: "repeat(2, minmax(0, 1fr))",
                          xs: "1fr",
                        },
                      }}
                    >
                      {summaryCards.map((card) => (
                        <Box key={card.label}>
                          <Card sx={{ height: "100%" }}>
                            <CardContent>
                              <Stack direction="row" spacing={1.5}>
                                {card.icon}
                                <Box>
                                  <Typography color="text.secondary" variant="caption">
                                    {card.label}
                                  </Typography>
                                  <Typography fontWeight={900} sx={{ overflowWrap: "anywhere" }} variant="h6">
                                    {card.value}
                                  </Typography>
                                </Box>
                              </Stack>
                            </CardContent>
                          </Card>
                        </Box>
                      ))}
                    </Box>

                    <Card>
                      <CardContent>
                        <Stack spacing={2}>
                          <Box>
                            <Typography variant="h2">Визуальные артефакты</Typography>
                          </Box>
                          <Box
                            sx={{
                              display: "grid",
                              gap: 2,
                              gridTemplateColumns: {
                                lg: "repeat(3, minmax(0, 1fr))",
                                sm: "repeat(2, minmax(0, 1fr))",
                                xs: "1fr",
                              },
                            }}
                          >
                            {visualArtifacts.map((artifact) => (
                              <Box key={artifact.key}>
                                <Card variant="outlined">
                                  <CardMedia
                                    alt={artifact.label}
                                    component="img"
                                    image={artifact.src}
                                    sx={{ bgcolor: "grey.100", height: 240, objectFit: "contain" }}
                                  />
                                  <CardContent sx={{ py: 1.5 }}>
                                    <Typography fontWeight={800} variant="body2">
                                      {artifact.label}
                                    </Typography>
                                  </CardContent>
                                </Card>
                              </Box>
                            ))}
                          </Box>
                        </Stack>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardContent>
                        <Stack spacing={2}>
                          <Box>
                            <Typography variant="h2">Морфология ядра</Typography>
                          </Box>
                          <Box
                            sx={{
                              display: "grid",
                              gap: 2,
                              gridTemplateColumns: {
                                md: "repeat(4, minmax(0, 1fr))",
                                sm: "repeat(2, minmax(0, 1fr))",
                                xs: "1fr",
                              },
                            }}
                          >
                            {[
                              ["Округлость", result.features.nucleus_circularity],
                              ["Solidity", result.features.nucleus_solidity],
                              ["Соотношение сторон", result.features.nucleus_aspect_ratio],
                              ["Доля маски", result.features.mask_foreground_fraction],
                            ].map(([label, value]) => (
                              <Box key={label}>
                                <Paper sx={{ p: 2 }} variant="outlined">
                                  <Typography color="text.secondary" variant="caption">
                                    {label}
                                  </Typography>
                                  <Typography fontWeight={900} variant="h6">
                                    {formatValue(value)}
                                  </Typography>
                                </Paper>
                              </Box>
                            ))}
                          </Box>
                          <Alert severity="info">{result.classification.reason}</Alert>
                        </Stack>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardContent>
                        <Stack spacing={2}>
                          <Box>
                            <Typography variant="h2">Детали анализа</Typography>
                          </Box>
                          <Tabs
                            onChange={(_, value) => setDetailsTab(value)}
                            scrollButtons="auto"
                            value={detailsTab}
                            variant="scrollable"
                          >
                            <Tab icon={<QueryStatsOutlinedIcon />} iconPosition="start" label="Признаки" value="features" />
                            <Tab icon={<FactCheckOutlinedIcon />} iconPosition="start" label="Классификация" value="classification" />
                            <Tab icon={<DataObjectOutlinedIcon />} iconPosition="start" label="Metadata" value="metadata" />
                            <Tab icon={<ArticleOutlinedIcon />} iconPosition="start" label="Артефакты" value="artifacts" />
                          </Tabs>
                          <Divider />

                          {detailsTab === "features" && (
                            <DetailTable rows={entriesToRows(result.features, featureLabels)} />
                          )}
                          {detailsTab === "classification" && (
                            <DetailTable
                              rows={[
                                { key: "label", name: "Класс", value: result.classification.label },
                                { key: "reason", name: "Обоснование", value: result.classification.reason },
                              ]}
                            />
                          )}
                          {detailsTab === "metadata" && (
                            <DetailTable rows={entriesToRows(result.metadata, metadataLabels)} />
                          )}
                          {detailsTab === "artifacts" && (
                            <TableContainer component={Paper} variant="outlined">
                              <Table size="small">
                                <TableHead>
                                  <TableRow>
                                    <TableCell>Артефакт</TableCell>
                                    <TableCell>Путь</TableCell>
                                    <TableCell align="right">Ссылка</TableCell>
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  {artifactRows.map((row) => (
                                    <TableRow key={row.key}>
                                      <TableCell sx={{ fontWeight: 700 }}>{row.name}</TableCell>
                                      <TableCell sx={{ overflowWrap: "anywhere" }}>{row.value}</TableCell>
                                      <TableCell align="right">
                                        <Link href={row.href} target="_blank" underline="hover">
                                          Открыть
                                        </Link>
                                      </TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          )}
                        </Stack>
                      </CardContent>
                    </Card>
                  </>
                ) : (
                  <Paper sx={{ display: "grid", minHeight: 460, placeItems: "center", p: 4 }} variant="outlined">
                    <Stack spacing={1.5} sx={{ alignItems: "center", maxWidth: 560, textAlign: "center" }}>
                      <ImageSearchOutlinedIcon color="primary" sx={{ fontSize: 48 }} />
                      <Typography fontWeight={900} variant="h5">
                        Ожидается изображение
                      </Typography>
                      <Typography color="text.secondary">
                        После запуска здесь появятся маски, overlay, классификация и табличные детали анализа.
                      </Typography>
                    </Stack>
                  </Paper>
                )}
              </Stack>
            </Box>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

createRoot(document.getElementById("root")).render(<App />);
